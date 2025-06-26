# train-test-val-split.py

import json
import os
from datasets import load_dataset
from tqdm import tqdm
import random

def categorize_and_split_combined_jsonl(
    inference_file_path,
    output_base_path,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    random_seed=42
):
    if not (train_ratio + val_ratio + test_ratio == 1.0):
        print("Error: Rasio pelatihan, validasi, dan pengujian harus berjumlah 1.0.")
        return

    random.seed(random_seed)

    print("Langkah 1: Memuat dataset asli dari Hugging Face untuk referensi instruksi dan output asli...")

    try:
        original_codealpaca_dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
        codealpaca_map = {item["instruction"].strip(): item["output"].strip() for item in original_codealpaca_dataset}
        print(f"  - {len(codealpaca_map)} instruksi unik dimuat dari CodeAlpaca.")
    except Exception as e:
        print(f"Error memuat CodeAlpaca: {e}. Pastikan Anda memiliki koneksi internet.")
        return

    try:
        original_gsm8k_dataset = load_dataset("openai/gsm8k", "main", split="train")
        gsm8k_map = {item["question"].strip(): item["answer"].strip() for item in original_gsm8k_dataset}
        print(f"  - {len(gsm8k_map)} pertanyaan unik dimuat dari GSM8K.")
    except Exception as e:
        print(f"Error memuat GSM8K: {e}. Pastikan Anda memiliki koneksi internet.")
        return

    print("\nLangkah 2: Memproses file inferensi dan menambahkan kategori serta expected_output...")

    all_categorized_samples = []
    processed_count = 0
    codealpaca_categorized_count = 0
    gsm8k_categorized_count = 0
    unmatched_count = 0
    ambiguous_count = 0

    output_dir = os.path.dirname(output_base_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        with open(inference_file_path, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
            for line in tqdm(lines, desc="Mengategorikan Sampel"):
                try:
                    data = json.loads(line)
                    instruction = data.get("instruction", "").strip()
                    
                    current_category = "unmatched"
                    current_expected_output = ""

                    if not instruction:
                        current_category = "unspecified_empty_instruction" 
                    else:
                        is_codealpaca = instruction in codealpaca_map
                        is_gsm8k = instruction in gsm8k_map

                        if is_codealpaca and not is_gsm8k:
                            current_category = "codealpaca"
                            current_expected_output = codealpaca_map[instruction]
                            codealpaca_categorized_count += 1
                        elif is_gsm8k and not is_codealpaca:
                            current_category = "gsm8k"
                            current_expected_output = gsm8k_map[instruction]
                            gsm8k_categorized_count += 1
                        elif is_codealpaca and is_gsm8k:
                            current_category = "ambiguous"
                            ambiguous_count += 1
                            current_expected_output = codealpaca_map[instruction] 
                        else:
                            current_category = "unmatched"
                            unmatched_count += 1
                    
                    data["category"] = current_category
                    data["expected_output"] = current_expected_output
                    
                    all_categorized_samples.append(data)
                    processed_count += 1

                except json.JSONDecodeError as e:
                    print(f"Error mendecode JSON pada baris ke-{processed_count + 1}: {e}. Melewatkan baris ini.")
                except Exception as e:
                    print(f"Error tidak terduga pada baris ke-{processed_count + 1}: {e}. Melewatkan baris ini.")

    except FileNotFoundError:
        print(f"Error: File '{inference_file_path}' tidak ditemukan. Pastikan path sudah benar.")
        return
    except Exception as e:
        print(f"Terjadi error saat membaca/menulis file: {e}")
        return

    print(f"\nLangkah 3: Membagi data secara stratifikasi menjadi set pelatihan, validasi, dan pengujian...")

    codealpaca_samples = [s for s in all_categorized_samples if s["category"] == "codealpaca"]
    gsm8k_samples = [s for s in all_categorized_samples if s["category"] == "gsm8k"]
    other_samples = [s for s in all_categorized_samples if s["category"] not in ["codealpaca", "gsm8k"]]

    random.shuffle(codealpaca_samples)
    random.shuffle(gsm8k_samples)
    random.shuffle(other_samples) # Shuffle any unmatched/ambiguous samples too

    train_data = []
    val_data = []
    test_data = []

    # Function to split a list proportionally
    def split_list_stratified(data_list, train_r, val_r, test_r):
        total = len(data_list)
        train_s = int(total * train_r)
        val_s = int(total * val_r)
        
        train = data_list[:train_s]
        val = data_list[train_s : train_s + val_s]
        test = data_list[train_s + val_s :]
        return train, val, test

    # Split each category and combine
    
    train_code, val_code, test_code = split_list_stratified(codealpaca_samples, train_ratio, val_ratio, test_ratio)
    train_gsm, val_gsm, test_gsm = split_list_stratified(gsm8k_samples, train_ratio, val_ratio, test_ratio)
    train_other, val_other, test_other = split_list_stratified(other_samples, train_ratio, val_ratio, test_ratio)

    train_data.extend(train_code)
    train_data.extend(train_gsm)
    train_data.extend(train_other)

    val_data.extend(val_code)
    val_data.extend(val_gsm)
    val_data.extend(val_other)

    test_data.extend(test_code)
    test_data.extend(test_gsm)
    test_data.extend(test_other)

    random.shuffle(train_data) # Shuffle combined train data
    random.shuffle(val_data)   # Shuffle combined val data
    random.shuffle(test_data)  # Shuffle combined test data

    def save_jsonl(data, file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    train_output_path = f"{output_base_path}_train.jsonl"
    val_output_path = f"{output_base_path}_val.jsonl"
    test_output_path = f"{output_base_path}_test.jsonl"

    save_jsonl(train_data, train_output_path)
    save_jsonl(val_data, val_output_path)
    save_jsonl(test_data, test_output_path)

    print(f"Dataset berhasil dibagi dan disimpan:")
    print(f"  - Pelatihan: {len(train_data)} sampel -> '{train_output_path}'")
    print(f"  - Validasi: {len(val_data)} sampel -> '{val_output_path}'")
    print(f"  - Pengujian: {len(test_data)} sampel -> '{test_output_path}'")

    print(f"\nProses kategorisasi dan pembagian selesai.")
    print(f"Total sampel diproses: {processed_count}")
    print(f"Sampel 'codealpaca' teridentifikasi: {codealpaca_categorized_count}")
    print(f"Sampel 'gsm8k' teridentifikasi: {gsm8k_categorized_count}")
    if ambiguous_count > 0:
        print(f"Sampel 'ambiguous' (ada di kedua dataset): {ambiguous_count}")
    if unmatched_count > 0:
        print(f"Sampel 'unmatched' (tidak cocok dengan dataset asli): {unmatched_count}")

if __name__ == "__main__":
    input_inference_jsonl = "Dataset/devstral_inference.jsonl" # Ganti dengan path file inferensi Anda
    output_base_path = "./Dataset/categorized_split_data_stratified" # Nama base path baru

    categorize_and_split_combined_jsonl(
        inference_file_path=input_inference_jsonl,
        output_base_path=output_base_path,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_seed=42
    )
