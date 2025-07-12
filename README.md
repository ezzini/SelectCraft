# SelectCraft 🧠🔍
**Domain-Specific Text-to-SQL Dataset Generation Framework**

SelectCraft is a framework for **generating high-quality, domain-specific Text-to-SQL datasets**. It helps bridge the gap between natural language and SQL queries for specialized applications, making it easier to train and evaluate Large Language Models in targeted domains (e.g., healthcare, finance, e-commerce, etc.).

---

## 🔧 Features

- 📊 **Schema-aware generation**: Tailors text-SQL pairs based on your domain-specific database schema.
- 💬 **Controlled natural language generation**: Uses LLMs to ensure diverse and accurate NL questions.
- ✅ **SQL validation**: Ensures generated queries are syntactically and semantically valid.

---


### Usage Instructions

- clone the repo and after installing the necessary dependencies:
```
pip -r install requirements.txt
```
- execute the file using :  
```
python sql_data_generation.py --dataset_size 1000 --number_of_paraphrases 4 --output_file "batch.csv"
```

### 🧪 Example
For a healthcare schema, the tool might generate:

NL: "What are the names of patients diagnosed with diabetes in 2025?"

SQL:

```sql
SELECT name FROM patients
WHERE diagnosis = 'diabetes'
AND year = '2025';
```

### 📈 Use Cases

- Fine-tuning or benchmarking Text-to-SQL models (e.g., StarCoder, CodeLlama, QwenCoder)
- Creating synthetic datasets for low-resource domains
- Evaluating semantic parsing in specialized industries
- Prototyping natural language interfaces for custom databases

### How does it work?

![approach_page-0001](https://github.com/user-attachments/assets/ff5d10a3-4dea-4685-8103-663109d7e9ff)

