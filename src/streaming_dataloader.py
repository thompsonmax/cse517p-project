import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, get_dataset_config_names, IterableDataset, interleave_datasets
from transformers import DataCollatorWithPadding # Or your specific model
import torch.optim as optim
import torch.nn as nn
import logging
import hyperparams
import string
import unicodedata
import random
import time

DATASET_NAME = 'wikimedia/wikipedia'

DESIRED_TOP_LANGUAGES = [
        'en',  # English
        'de',  # German
        'fr',  # French
        'es',  # Spanish
        'ru',  # Russian
        'ja',  # Japanese
        'it',  # Italian
        'zh',  # Chinese
        'pt',  # Portuguese
        'ar',  # Arabic
        'pl',  # Polish
        'nl',  # Dutch
        # 'ceb', # Cebuano (often large by article count due to bots, content quality varies)
        # Uncomment to get better language coverage
        # 'sv',  # Swedish
        # 'hin', # Hindi
        # 'tr',  # Turkish
        # 'vi',  # Vietnamese
        # 'id',  # Indonesian
        # 'ko',  # Korean
        # 'fi',  # Finnish
        # 'no',  # Norwegian
        # 'da',  # Danish
        # 'he',  # Hebrew
        # 'uk',  # Ukrainian
        # 'ro',  # Romanian
        # 'cs',  # Czech
        # 'hu',  # Hungarian
        # 'bg',  # Bulgarian
        # 'el',  # Greek
        # 'th',  # Thai
        # 'fa',  # Persian
        # 'sk',  # Slovak
        # 'lt',  # Lithuanian
        # 'sl',  # Slovenian
        # 'hr',  # Croatian
        # 'et',  # Estonian
        # 'lv',  # Latvian
        # 'sr',  # Serbian
        # 'az',  # Azerbaijani
        # 'mk',  # Macedonian
        # 'ms',  # Malay
        # 'ca',  # Catalan
        # 'gl',  # Galician
        # 'eu',  # Basque
        # 'is',  # Icelandic
        # 'mt',  # Maltese
        # 'af',  # Afrikaans
        # 'sw',  # Swahili
        # 'tl',  # Tagalog
        # 'bn',  # Bengali
        # 'pa',  # Punjabi
        # 'gu',  # Gujarati
        # 'ta',  # Tamil
        # 'te',  # Telugu
        # 'kn',  # Kannada
        # 'ml',  # Malayalam
        # 'mr',  # Marathi
        # 'or',  # Odia
        # 'si',  # Sinhala
        # 'ur',  # Urdu
        # 'my',  # Burmese
        # 'km',  # Khmer
        # 'lo',  # Lao
        # 'ne',  # Nepali
        # 'am',  # Amharic
        # 'yo',  # Yoruba
        # 'ig',  # Igbo
        # 'zu',  # Zulu
        # 'xh',  # Xhosa
        # 'af',  # Afrikaans
        # 'ht',  # Haitian Creole
        # 'cy',  # Welsh
        # 'ga',  # Irish
        # 'sq',  # Albanian
        # 'bs',  # Bosnian
        # 'mk',  # Macedonian
        # 'mn',  # Mongolian
        # 'uz',  # Uzbek
        # 'tg',  # Tajik
        # 'ky',  # Kyrgyz
        # 'tk',  # Turkmen
        # 'az',  # Azerbaijani
        # 'ba',  # Bashkir
        # 'tt',  # Tatar
        # 'ky',  # Kyrgyz
        # 'gl',  # Galician
        # 'ast', # Asturian
    ]
TARGET_LANG_COUNT = len(DESIRED_TOP_LANGUAGES)
# DATASET_CONFIG = '20231101.ab'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_streaming_dataloader(vocab2idx: dict) -> DataLoader:

    logger.info(f"Fetching available configurations for {DATASET_NAME}...")
    try:
        # For wikimedia/wikipedia, trust_remote_code might be needed if it uses a custom loading script
        available_configurations = get_dataset_config_names(DATASET_NAME, trust_remote_code=True)
    except Exception as e:
        raise ValueError(f"Failed to fetch dataset configurations for {DATASET_NAME}. Please check the dataset name or your internet connection. Error: {e}")

    logger.info(f"Found {len(available_configurations)} raw configurations.")

    logger.info(f"First few available configurations: {available_configurations[:10]}")
    logger.info(f"A '20231101.en' style config generally means DATE.LANG_CODE")

    # Find the latest available dump for ALL languages first
    latest_configs_for_all_langs = {}
    for config_name in available_configurations:
        parts = config_name.split('.', 1)
        if len(parts) == 2:
            date_str, lang_code = parts
            # Normalize lang_code (e.g., if there are variants like zh-hans, zh-hant, map to 'zh' if desired)
            # For wikimedia/wikipedia, lang codes are usually simple e.g. 'zh', 'en'
            if lang_code not in latest_configs_for_all_langs or date_str > latest_configs_for_all_langs[lang_code]['date']:
                latest_configs_for_all_langs[lang_code] = {'date': date_str, 'config': config_name, 'lang': lang_code}
        else:
            # Handle configs that don't fit the date.lang pattern (e.g., just 'simple')
            # These are less common for the main language dumps of wikimedia/wikipedia
            lang_code = config_name
            if lang_code not in latest_configs_for_all_langs: # Avoid overwriting if a dated version is preferred
                latest_configs_for_all_langs[lang_code] = {'date': 'nodate', 'config': config_name, 'lang': lang_code}

    logger.info(f"Identified latest dumps for {len(latest_configs_for_all_langs)} unique language codes.")

    # Now, filter these latest dumps to include only our desired top languages
    # and take up to TARGET_LANG_COUNT
    processed_languages_to_load = []
    loaded_lang_codes = set()

    for lang_code_to_check in DESIRED_TOP_LANGUAGES:
        if lang_code_to_check in latest_configs_for_all_langs:
            if lang_code_to_check not in loaded_lang_codes: # Ensure unique languages
                processed_languages_to_load.append(latest_configs_for_all_langs[lang_code_to_check]['config'])
                loaded_lang_codes.add(lang_code_to_check)
                if len(processed_languages_to_load) >= TARGET_LANG_COUNT:
                    break # Stop once we have enough languages


    logger.info(f"Will attempt to load {len(processed_languages_to_load)} unique language configurations (latest available dump).")
    logger.info(f"Example configurations to be loaded: {processed_languages_to_load}")


    # 2. Load and process each language dataset (using streaming)
    all_language_datasets_streamed = {}

    for lang_config_name in processed_languages_to_load:
        # Ensure we only try to load valid configurations obtained
        try:
            logger.info(f"Attempting to load '{DATASET_NAME}' with configuration '{lang_config_name}' in streaming mode...")
            # The wikipedia dataset typically has a 'train' split.
            # Some datasets might require specifying a specific version or trust_remote_code=True
            dataset_stream = load_dataset(
                DATASET_NAME,
                name=lang_config_name,
                streaming=True,
                split="train",
            )
            all_language_datasets_streamed[lang_config_name] = dataset_stream
            logger.info(f"Successfully initiated stream for {lang_config_name}.")
            # Sleep for 5 seconds to avoid 429s from the API
            time.sleep(5)


        except Exception as e:
            logger.error(f"Could not load language {lang_config_name}: {e}")
            raise ValueError(f"Failed to load dataset for {lang_config_name}. Please check the configuration name or your internet connection. Error: {e}")
            # logger.warning(f"Skipping language: {lang_config_name}")

    logger.info(f"\nSuccessfully set up streams for {len(all_language_datasets_streamed)} languages.")
    logger.info("You can now iterate through `all_language_datasets_streamed` dictionary,")
    logger.info("where keys are language configuration names and values are iterable dataset streams.")

    interleaved_dataset = interleave_datasets(list(all_language_datasets_streamed.values()))
    logger.info("Interleaved dataset created successfully.")

    # Map dataset to unicode code points and then indices of the vocab using vocab2idx
    logger.info("Mapping dataset to vocabulary indices...")
    def map_to_vocab_indices(example):
        # print(f"Mapping example: {example['text'][:10]}...")  # Print first 10 characters of the text
        input_ids = []
        for char in unicodedata.normalize('NFC', example['text']):
            # print(char)
            code_point = ord(char)
            # print(ord(char))
            idx = vocab2idx.get(code_point, hyperparams.UNK_CHAR_IDX)
            input_ids.append(idx)

        output_ids = {
            'ids': input_ids
        }
        # print(f"Mapped input_ids: {output_ids['input_ids'][:10]}...")  # Print first 10 mapped IDs
        return output_ids
    interleaved_dataset = interleaved_dataset.map(
        map_to_vocab_indices
    )
    logger.info("Mapped dataset to vocabulary indices successfully.")

    SEQ_LENGTH = hyperparams.SEQ_LENGTH

    def _pad_sequence_helper(sequence: list, max_len: int, padding_value: int) -> list:
        padding_needed = max_len - len(sequence)
        # Ensure padding_needed is not negative, though slicing should prevent oversized sequences.
        if padding_needed > 0:
            return sequence + ([padding_value] * padding_needed)
        return sequence[:max_len] # Ensure sequence is not longer than max_len, though logic aims for exact

    def create_source_target_next_token_pairs(example):
        # Randomly samples a sequence pair from the document
        ids = example['ids']
        
        source_chunk, target_chunk = None, None 
        
        padding_id = hyperparams.PADDING_CHAR_IDX

        n = len(ids)

        num_complete_pair_starts = n - SEQ_LENGTH

        if num_complete_pair_starts > 0:
            start_idx = random.randrange(num_complete_pair_starts) 
            
            source_chunk = ids[start_idx : start_idx + SEQ_LENGTH]
            target_chunk = ids[start_idx + 1 : start_idx + 1 + SEQ_LENGTH]
        else:
            source_prefix = ids[:SEQ_LENGTH] 
            source_chunk = _pad_sequence_helper(source_prefix, SEQ_LENGTH, padding_id)
            
            target_prefix = ids[1 : SEQ_LENGTH + 1] 
            target_chunk = _pad_sequence_helper(target_prefix, SEQ_LENGTH, padding_id)
            
        return {'input_ids': source_chunk, 'target_ids': target_chunk}

    logger.info("Creating source-target next token pairs...")
    paired_dataset = interleaved_dataset.map(
        create_source_target_next_token_pairs,
    )
    logger.info("Source-target pairs created successfully.")

    # Update the reference to the dataset
    interleaved_dataset = paired_dataset

    logger.info("Shuffling dataset...")
    interleaved_dataset = interleaved_dataset.shuffle(seed=42)
    logger.info("Dataset shuffled successfully.")


    logger.info("Creating DataLoader...")
    return DataLoader(
        interleaved_dataset,
        batch_size=hyperparams.BATCH_SIZE,
        collate_fn=collator,
        num_workers=0,
    )

def collator(batch):
    input_id_lists = [sample['input_ids'] for sample in batch]
    # for input_ids in input_id_lists:
        # print(f"Input ID length: {len(input_ids)}")
    batched_input_ids = torch.LongTensor(input_id_lists)

    target_id_lists = [sample['target_ids'] for sample in batch]
    batched_target_ids = torch.LongTensor(target_id_lists)

    return_dict = {
        'input_ids': batched_input_ids,
        'target_ids': batched_target_ids
    }
    
    return return_dict

if __name__ == "__main__":
    example_vocab = {}
    character_sets = [
        string.ascii_lowercase,  # 'abcdefghijklmnopqrstuvwxyz'
        string.ascii_uppercase,  # 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        string.digits           # '0123456789'
    ]

    next_char_idx = 0
    for char_set in character_sets:
        for char in char_set:
            code_point = ord(char)
            example_vocab[code_point] = next_char_idx
            next_char_idx += 1

    print(f"Example vocab size: {len(example_vocab)}")
    
    # Create the streaming dataloader
    dataloader = create_streaming_dataloader(example_vocab)
    # Example usage of the dataloader
    i = 0
    for batch in dataloader:
        # Process your batch here
        print(batch)  # This will print the input_ids for each batch
        if i > 10:
            break
        i += 1
    print("Finished processing batches.")