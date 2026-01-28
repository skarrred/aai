#p10
#Use Python libraries such as GPT-2 or textgenrnn to train generative models on a corpus of text data and generate new text based on the patterns it has learned.( !pip install textgenrnn)

!pip install git+https://github.com/minimaxir/textgenrnn.git

import tensorflow as tf
from transformers import pipeline

generator = pipeline('text-generation', model='gpt2')

input_text = " Once upon a time in a faraway land, "

# Generate text
num_generated_sequences = 1
max_length_of_generated_text = 100
temperature_for_generation = 0.7
top_k_for_generation = 50

print("\nGenerated Text:\n")

generated_text = generator(
    input_text,
    max_length=max_length_of_generated_text,
    num_return_sequences=num_generated_sequences,
    temperature=temperature_for_generation,
    top_k=top_k_for_generation
)

for i, sample in enumerate(generated_text):
    print(f"Sequence {i+1}: {sample['generated_text']}")


