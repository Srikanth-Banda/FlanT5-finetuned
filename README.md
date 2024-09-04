
# **FlanT5 Fine-Tuning for Python Code Generation Using PEFT/LoRA**

## **Project Overview**

This repository contains the code and resources for fine-tuning the **FlanT5-base** model for Python code generation using the **Parameter Efficient Fine-Tuning (PEFT)** technique. The project demonstrates how LoRA (Low-Rank Adaptation) can be used to efficiently adapt a large language model for specialized code generation tasks, reducing memory and computational overhead.

## **Project Structure**
- **Data**: The model was trained on the **18k Python Code Instructions dataset** from Alpaca.
- **Model**: Pre-trained **FlanT5-base** was selected from Hugging Face and further fine-tuned to generate Python code.
- **Techniques Used**: 
  - **PEFT (LoRA)** for efficient training.
  - **PyTorch** and **Transformers** for model training.
  - **ROUGE Score** for evaluating performance.
  
## **Training Process**

1. **Preprocessing**: The dataset was preprocessed, and inputs were tokenized to match the model's requirements.
2. **Fine-Tuning**: A PEFT approach with LoRA was employed, significantly reducing the number of trainable parameters (~1.41%).
3. **Evaluation**: The fine-tuned model was compared against the base model using **ROUGE metrics**, showing substantial improvement in code generation quality.

## **Key Components**

- **FlanT5-base Model**: A powerful instruction-following model that provides a solid foundation for generating Python code.
- **PEFT/LoRA**: An efficient way to fine-tune models, enabling resource-constrained environments to fine-tune large models without extensive computational resources.
- **Hugging Face Datasets**: Leveraged Hugging Face datasets to handle and load large datasets efficiently.

## **Results**

- The fine-tuned model outperformed the original FlanT5-base on the task of generating Python code, making it highly suitable for tasks like Python script generation and code completion.
- **ROUGE Metrics**:
  - Substantial improvement across various Rouge metrics, with a significant boost in performance after fine-tuning using PEFT.

## **How to Run the Project**

1. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Pre-trained Model**: Clone the model from Hugging Face’s repository.
   ```bash
   git clone https://huggingface.co/google/flan-t5-base
   ```

3. **Run Fine-Tuning**: The notebook includes scripts to fine-tune the model on your own dataset.
   
4. **Generate Code**: Use the fine-tuned model for inference.
   ```python
   from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

   tokenizer = AutoTokenizer.from_pretrained('path_to_finetuned_model')
   model = AutoModelForSeq2SeqLM.from_pretrained('path_to_finetuned_model')

   prompt = "Write a Python function to calculate the sum of a list."
   inputs = tokenizer(prompt, return_tensors='pt').input_ids
   outputs = model.generate(inputs)
   print(tokenizer.decode(outputs[0]))
   ```

## **Contributions**
Feel free to contribute to this project by submitting pull requests, reporting bugs, or suggesting improvements. Your feedback is always welcome!

## **License**
This project is licensed under the MIT License.
