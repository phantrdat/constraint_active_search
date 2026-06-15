import torch
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, QED
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Crippen import MolLogP
from tqdm import tqdm
import os
import glob

# Attempt to import sascorer, provide a fallback if not available.
from rdkit.Contrib.SA_Score import sascorer
# except ImportError:
#     sascorer = None
#     print("RDKit SA_Score module not found. SA scores will be zero.")

OBJ_COLS_NAMES = ["objective_1_docking", "objective_2_esol", "objective_3_sa_score", "objective_4_qed", "objective_5_tanimoto"]

def _calculate_objectives_for_chunk(chunk_df, top_k_tanimoto_mols_fps, fpgen):
    """
    Calculates the 5 objectives for a given chunk of molecules.
    Assumes 'mol' and 'docking_score' columns are present in chunk_df.
    """
    # Obj 1: Docking Score (higher is better, so we negate the score)
    chunk_df['objective_1_docking'] = -chunk_df['docking_score'].values

    # Obj 2: ESOL (predicted solubility, using MolLogP as proxy)
    chunk_df['objective_2_esol'] = [MolLogP(mol) for mol in chunk_df['mol']]

    # Obj 3: SA Score (synthetic accessibility, lower is better, so we negate)
    chunk_df['objective_3_sa_score'] = [-sascorer.calculateScore(mol) for mol in chunk_df['mol']] if sascorer else np.zeros(len(chunk_df))

    # Obj 4: QED (drug-likeness)
    chunk_df['objective_4_qed'] = [QED.qed(mol) for mol in chunk_df['mol']]

    # Obj 5: Tanimoto Similarity to Top K
    # Calculate fingerprints for the current chunk
    chunk_fps = [fpgen.GetFingerprint(m) for m in chunk_df['mol']]
    chunk_df['objective_5_tanimoto'] = [max(DataStructs.BulkTanimotoSimilarity(fp, top_k_tanimoto_mols_fps)) for fp in chunk_fps]

    return chunk_df

def preprocess_drug_data_in_chunks(target_name, data_dir, top_k_tanimoto, num_data_parts, data_part_idx, chunk_size=10000):
    """
    Loads molecule data in chunks, generates RDKit descriptors, and calculates the 5 objectives.
    Caches the complete processed DataFrame to a .pkl file to speed up subsequent initializations.
    This function is designed to be memory efficient by processing data in chunks.
    """
    cache_dir = os.path.join(data_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    df_cache_file = os.path.join(cache_dir, f"{target_name}_part{data_part_idx}_of_{num_data_parts}.pkl")

    if os.path.exists(df_cache_file):
        print(f"Preprocessed data already exists at {df_cache_file}. Skipping preprocessing.")
        return df_cache_file

    print(f"Starting full data preprocessing for {target_name} (part {data_part_idx} of {num_data_parts})...")
    target_map = {
        "3CLPro": "COVIDRec/3CLPro_7BQY_A_1_F",
        "RTCB": "CancerRep/RTCB",
        "6T2W": "CancerRep/6t2w-4-amino-2-methyl-5-pyrimidinol",
        "WRN": "CancerRep/wrn-NSC216260"
    }
    if target_name not in target_map:
        raise NotImplementedError(f"Target {target_name} data path not specified in target_map.")

    target_path_pattern = os.path.join(data_dir, target_map[target_name], "SMILES*.csv")
    smi_files = glob.glob(target_path_pattern)
    if not smi_files:
        raise FileNotFoundError(f"No SMILES CSV files found at {target_path_pattern}")

    input_csv_path = smi_files[0]

    # --- Step 1: Load docking scores to determine top K for Tanimoto ---
    # This step loads a subset of the data to find the top K molecules for Tanimoto similarity.
    # For extremely large datasets, this part might still be memory intensive and require
    # a more advanced streaming/external sorting approach.
    print(f"Loading all molecules to determine top K for Tanimoto similarity...")
    full_df_for_top_k = pd.read_csv(input_csv_path, usecols=["SMILES", "DockingScore"])
    full_df_for_top_k.rename(columns={"SMILES": "smiles", "DockingScore": "docking_score"}, inplace=True)

    # Find indices of top K molecules based on original (lowest) docking scores from the full dataset
    temp_docking_series = pd.Series(full_df_for_top_k['docking_score'])
    top_k_indices_global = temp_docking_series.nsmallest(top_k_tanimoto).index
    top_k_smiles_global = full_df_for_top_k.loc[top_k_indices_global, 'smiles'].tolist()
    top_k_mols_global = [Chem.MolFromSmiles(smi) for smi in tqdm(top_k_smiles_global, desc="Parsing Top K SMILES")]
    top_k_mols_global = [mol for mol in top_k_mols_global if mol is not None] # Filter out invalid mols
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    top_k_tanimoto_mols_fps = [fpgen.GetFingerprint(mol) for mol in top_k_mols_global]
    del full_df_for_top_k # Free up memory

    # --- Step 2: Process data in chunks ---
    processed_chunks = []
    
    # Determine the slice of the data this part should process
    total_mols_in_file = pd.read_csv(input_csv_path, usecols=["SMILES"]).shape[0] # Get actual total count
    mols_to_process_for_this_part = total_mols_in_file # Process the entire file, split into parts

    part_size = mols_to_process_for_this_part // num_data_parts
    start_row = data_part_idx * part_size
    end_row = start_row + part_size
    if data_part_idx == num_data_parts - 1: # Ensure the last part gets all remaining molecules
        end_row = mols_to_process_for_this_part

    print(f"Processing molecules from row {start_row} to {end_row-1} in chunks of {chunk_size}...")

    reader = pd.read_csv(input_csv_path, iterator=True, chunksize=chunk_size)
    current_row = 0
    
    for i, chunk in enumerate(reader):
        chunk_start_idx = current_row
        chunk_end_idx = current_row + len(chunk)

        # Skip chunks entirely before our part's start_row
        if chunk_end_idx <= start_row:
            current_row = chunk_end_idx
            continue
        
        # Stop if we've processed past our part's end_row
        if chunk_start_idx >= end_row:
            break

        # Adjust chunk to only include rows relevant to this part
        actual_chunk_start = max(start_row, chunk_start_idx)
        actual_chunk_end = min(end_row, chunk_end_idx)
        
        # Calculate local indices within the chunk
        local_start = actual_chunk_start - chunk_start_idx
        local_end = local_start + (actual_chunk_end - actual_chunk_start)
        
        if local_start >= local_end: # Empty chunk after slicing
            current_row = chunk_end_idx
            continue

        processed_chunk = chunk.iloc[local_start:local_end].copy()
        processed_chunk.rename(columns={"SMILES": "smiles", "DockingScore": "docking_score"}, inplace=True)

        print(f"  Processing chunk {i+1} (rows {actual_chunk_start}-{actual_chunk_end-1})...")

        # Generate RDKit molecules
        processed_chunk['mol'] = [Chem.MolFromSmiles(smi) for smi in processed_chunk['smiles']]
        processed_chunk.dropna(subset=['mol'], inplace=True)
        if processed_chunk.empty:
            current_row = chunk_end_idx
            continue

        # Generate RDKit descriptors
        desc_list = [d[0] for d in Descriptors._descList]
        descriptors_data = []
        for mol in tqdm(processed_chunk['mol'], desc="Calculating descriptors"):
            try:
                descriptors_data.append(Descriptors.CalcMolDescriptors(mol))
            except Exception: # Catch potential errors in descriptor calculation for some molecules
                descriptors_data.append({d: np.nan for d in desc_list}) # Fill with NaN
        
        desc_df = pd.DataFrame(descriptors_data, columns=desc_list, index=processed_chunk.index)
        desc_df = desc_df.replace([np.inf, -np.inf], np.nan)
        

        # We drop 'qed' and 'MolLogP' because they are used as objectives or are redundant for this target.
        cols_to_remove = ['qed', 'MolLogP']
        desc_df.drop(columns=[c for c in cols_to_remove if c in desc_df.columns], inplace=True)
        
        print(f"      Dropped {cols_to_remove} from features to avoid objective leakage.")
        
        # Drop rows from processed_chunk where descriptor calculation failed or resulted in NaN/inf
        # This ensures alignment between processed_chunk and desc_df
        valid_desc_rows = desc_df.notna().all(axis=1)
        processed_chunk = processed_chunk[valid_desc_rows].reset_index(drop=True)
        desc_df = desc_df[valid_desc_rows].reset_index(drop=True)

        # Keep only columns that are valid for all molecules in the subset
        desc_df.dropna(axis=1, how='any', inplace=True)
        
        # Prefix descriptor columns
        desc_df.columns = [f"descriptor_{col}" for col in desc_df.columns]

        # Merge processed_chunk and desc_df
        processed_chunk = pd.concat([processed_chunk, desc_df], axis=1)

        # Calculate objectives for the chunk
        processed_chunk = _calculate_objectives_for_chunk(processed_chunk, top_k_tanimoto_mols_fps, fpgen)
        
        # Drop the 'mol' column as it's not needed for final storage and not easily serializable
        processed_chunk.drop(columns=['mol'], inplace=True)
        processed_chunks.append(processed_chunk)
        
        current_row = chunk_end_idx

    if not processed_chunks:
        raise ValueError("No data processed. Check input parameters and data paths.")

    # Concatenate all processed chunks
    full_processed_df = pd.concat(processed_chunks, ignore_index=True)

    # # --- Step 3: Global Normalization of Objectives ---
    # print("Performing global normalization of objective columns to [0, 1]...")
    # obj_cols = [col for col in full_processed_df.columns if col.startswith('objective_')]
    
    # for col in obj_cols:
    #     min_val = full_processed_df[col].min()
    #     max_val = full_processed_df[col].max()
    #     # Add a small epsilon to avoid division by zero if all values are the same
    #     full_processed_df[col] = (full_processed_df[col] - min_val) / (max_val - min_val + 1e-6)

    print(f"Saving full preprocessed DataFrame to {df_cache_file}...")
    full_processed_df.to_pickle(df_cache_file)
    
    return df_cache_file

import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

def perform_feature_selection(cache_dir, target_name, num_data_parts, correlation_threshold=0.95):
    # 1. Load the dataset
    all_parts = []
    for i in range(0, num_data_parts):
            # Using a glob or formatted string to find the specific part file
        part_file = os.path.join(cache_dir, f"{target_name}_part{i}_of_{num_data_parts}.pkl")
            
        if os.path.exists(part_file):
            print(f"  Loading part {i}: {part_file}")
            part_df = pd.read_pickle(part_file)
            all_parts.append(part_df)
        else:
                # Optional: handle missing parts
            print(f"  Warning: Part {i} not found ({part_file}). Skipping.")

    if not all_parts:
        raise FileNotFoundError(f"No data parts found for {target_name} in {cache_dir}")

    # Concatenate all loaded dataframes
    df = pd.concat(all_parts, ignore_index=True)
    # Identify feature columns (descriptors) and target columns (objectives)
    feature_cols = [c for c in df.columns if c.startswith('descriptor_')]
    objective_cols = [c for c in df.columns if c.startswith('objective_')]
    
    print(f"Initial descriptor count: {len(feature_cols)}")
    X = df[feature_cols].copy()
    
    # 2. Handle Missing Values
    # Drop columns that are entirely NaN, and fill others with the median
    X = X.dropna(axis=1, how='all')
    X = X.fillna(X.median())
    print(f"Count after dropping all-NaN columns: {X.shape[1]}")
    
    # 3. Remove Constant Features (Zero Variance)
    # This removes descriptors that have the same value for every molecule
    selector = VarianceThreshold(threshold=0)
    selector.fit(X)
    X = X.iloc[:, selector.get_support()]
    print(f"Count after removing constant features: {X.shape[1]}")
    
    # 4. Remove Highly Correlated Features
    # Redundant features (e.g., different versions of Molecular Weight) 
    # can bias Gaussian Process models.
    corr_matrix = X.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # Find features with correlation greater than the threshold
    to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
    
    X = X.drop(columns=to_drop)
    print(f"Count after removing correlated features (> {correlation_threshold}): {X.shape[1]}")
    
    # 5. Finalize the cleaned dataset
    # Combine the SMILES, Objectives, and the selected Descriptors
    final_df = pd.concat([df[['smiles']], df[objective_cols], X], axis=1)
    
    return final_df




if __name__ == "__main__":
    # Example usage:
    for i in range(0, 1):
        preprocess_drug_data_in_chunks(
            target_name="3CLPro",
            data_dir="data/DrugImprover/data",
            top_k_tanimoto=1024,
            num_data_parts=50000,
            data_part_idx=i, chunk_size=20
        )
    
    # # Execute the selection
    # cleaned_data = perform_feature_selection(
    #     r'data/DrugImprover/data/cache/',
    #     target_name="3CLPro",
    #     num_data_parts=20,
    #     correlation_threshold=0.99)
    # feature_cols = [col for col in cleaned_data.columns if col.startswith('descriptor_')]
    
    # print(f"\nSelected {len(feature_cols)} features after preprocessing and feature selection.")
    # feature_cols = [col.replace('descriptor_', '') for col in feature_cols]
    # print(f"Selected features: {feature_cols}")
    # # Save the result
    # cleaned_data.to_pickle('data/DrugImprover/data/cache/3CLPro_selected_features.pkl')
    # print("\nFeature selection complete. Processed data saved to '3CLPro_selected_features.pkl'.")
    # cleaned_data.head(20).to_csv('data/DrugImprover/data/cache/3CLPro_selected_features.csv', index=False)
    