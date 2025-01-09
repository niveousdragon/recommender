from llama_cpp import Llama
import torch
import os
import pandas as pd
import random

from sympy.integrals.meijerint_doc import category
from translate import Translator
from preprocessing import preprocess_group, get_all_preprocessed_subcategories

translator = Translator(to_lang="en", from_lang='ru')

print(torch.cuda.is_available())

from llama_cpp import Llama

# Path to the .gguf model file
#model_path = "Meta-Llama-3.1-70B-Instruct-IQ2_XS.gguf"
model_path = 'Llama-3.3-70B-Instruct-Q6_K_L-00001-of-00002.gguf'

# Initialize the LLaMA model with GPU support
llm = Llama(
    model_path=model_path,
    n_gpu_layers=32,  # Adjust the number of layers to offload to the GPU
    n_threads=30,      # Number of CPU threads for preprocessing
    n_batch=2048,      # Batch size for token processing
    use_mmap=True,    # Memory mapping to reduce RAM usage
    use_mlock=True    # Lock model in RAM for performance
)

all_subcats, reverse_cat = get_all_preprocessed_subcategories(enable_stemming=False)

# Test the model
#prompt = "Top 50 products in the 'groceries' category are:"
#category = "Журналы и газеты"
#en_category = translator.translate(category)

#scs = random.sample(all_subcats[:], 25)
scs = all_subcats[:]
cs = [reverse_cat[sc] for sc in scs]
ilists = []
llm_df = pd.DataFrame()

for sc in scs:
    category = "".join(sc)
    prompt = f"10 самых популярных товаров, купленных в категории {category}, перечисленных через запятую, это:"
    output = llm(prompt, max_tokens=500, stop=["\n", "###"])
    #prompt2 = f"В категории {category} продаются следующие товары:"
    #prompt2 = f"25 most popular items bought in the category {en_category} are:"

    print('subcategory:', sc)
    print('big category:', reverse_cat[sc])
    print(f'answer:')
    print(output["choices"][0]["text"])

    #output2 = llm(prompt2, max_tokens=500, stop=["\n", "###"])
    try:
        itemlist = output["choices"][0]["text"].split(',')
    except:
        itemlist = []
    ilists.append(itemlist)

llm_df['subcategory'] = scs
llm_df['category'] = cs
llm_df['10 products'] = ilists

print(llm_df.head())
llm_df.to_excel('LLM test.xlsx')
# Print the output



