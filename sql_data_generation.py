import random, re, sys, argparse, itertools, torch, csv
from tqdm import tqdm
import pandas as pd

from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModelForSeq2SeqLM



#list of columns for each table : DB SCHEMA
columns = {
        'Transactions': ["transaction_id","timestamp_id","primary_contract_id","client_id","beneficiary_id","transaction_amount","amount_currency","product_family_code","is_fraudulent"],
        'Beneficiary': ["beneficiary_id","bank_branch_id","country_name","country_code"],
        'Source': ["primary_contract_id", "client_id", "counterparty_bank_branch_id", "counterparty_donor_id"],
        'Time': ["timestamp_id","week_number","day_number","hour_number","day_name","year","month_number"]
}

#list of tables' names
tables = list(columns.keys())

# List of possible operators
operators = ['=', '>', '<', '<>', 'IN', 'LIKE','>=', '<=','BETWEEN']
# string operators
soperators = ['=', '<>', 'IN', 'LIKE']
# code(ids) operators
coperators = ['=', '<>', 'IN']
# boolean operators
boperators = ['=', '<>']
# non string operators (float, int)
nsoperators = ['=', '>', '<', '<>','>=', '<=', 'BETWEEN']

# list of aggregation functions to use
aggregate_ops = ["COUNT", "SUM", "AVG", "MAX", "MIN"]


tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-wikiSQL-sql-to-en")
model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-wikiSQL-sql-to-en")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

t5_tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
t5_model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")


def main():
    parser = argparse.ArgumentParser(description='sql_generation')
    parser.add_argument('--dataset_size', dest='dataset_size', type=int, help='The number of queries to generate')
    parser.add_argument('--number_of_paraphrases', dest='number_of_paraphrases', type=int, help=' The number of paraphrases per query')
    parser.add_argument('--output_file', dest='output_file', type=str, help='The path to the output file')
    args = parser.parse_args()


    generate_batch(args.dataset_size, args.number_of_paraphrases, args.output_file)


def get2vals(col):
    i,j=0,0
    if col in ['bank_branch_id','transaction_id','beneficiary_id','primary_contract_id','client_id','counterparty_donor_id','counterparty_bank_branch_id','timestamp_id','product_family_code','transaction_amount']:
      i,j=1,100000
    elif col == 'year':
      i,j=2000,2024
    elif col=='month_number':
      i,j=1,12
    elif col=='week_number':
      i,j=1,7
    elif col=='hour_number':
      i,j=1,24
    elif col=='day_number':
      i,j=1,31
    if j!=0 and i!=0:
      fv=random.randint(i, j-3)    #fv:first value
      return [fv,random.randint(fv+2, j)]
    if col=='amount_currency':
      return ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'CNH', 'HKD', 'NZD']
    if col=='is_fraudulent':
      return ['Yes', 'No']
    if col=='day_name':
      return ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    if col=='country_name':
      return ['Luxembourg', 'USA', 'Canada', 'Germany', 'France', 'England','Belgium']
    if col=='country_code':
      return ['LU','US','CA','DE','FR','UK','BE']
    print("ERROR Col not defined:",col)

#to verify if we can process an aggregation function on this type or not (for example a sum or avg ...)
def iscat(col):
  if col in ['transaction_amount','year','month_number','week_number','day_number','hour_number']:
    return False
  return True

def coltype(col):
  if col in ['country_code','country_name','day_name','amount_currency']:
    return 's'
  elif col in ['bank_branch_id','transaction_id','beneficiary_id','primary_contract_id','client_id','counterparty_donor_id','counterparty_bank_branch_id','timestamp_id','product_family_code']:
    return 'c'
  elif col in ['is_fraudulent']:
    return 'b'
  else:
    return 'ns'

def get_op(col):
  ctype=coltype(col)
  if ctype=='s':
    return random.choice(soperators)
  if ctype=='b':
    return random.choice(boperators)
  if ctype=='ns':
    return random.choice(nsoperators)
  if ctype=='c':
    return random.choice(coperators)

# generate a random aggregation function based on the column type
def get_agg(col):
  ctype=iscat(col)
  if ctype:
    return "COUNT" # there are some function for playing with strings (concatenations ...) we can add those to
  return random.choice(aggregate_ops)

def generate_query(group=1):
  agg, join, where, group_by, try_negation, negation=(False,False,False,False,False,False)

  # the group number defines the complexity of the sql query that we want to generate
  if group>=1:
    where=True
  if group>=2:
    agg=True
  if group>=3:
    group_by=True
  if group>=4:
    join=True

  use_aggregate_func = agg
  use_join = join
  use_where = where
  use_group_by = group_by

  try_negation = True

  table1,table2='',''

  # if the group>=4 (complex query) chose 2 random tables to join
  if use_join:
      table1=tables[0]
      table2=random.choice(tables[1:])
      join_columns = list(set(columns[table1]).intersection(columns[table2]))
      join_column = join_columns[0]
  else:
      table1=random.choice(tables)
      join_column = None

  join_op= random.choices(["JOIN","INNER JOIN","LEFT JOIN","RIGHT JOIN","FULL OUTER JOIN"], weights=(47.58,25.95,23.85,1.98,0.64),k=1)[0] ## based on the Stack dataset statistics

  column_combination = random.sample([x for x in columns[table1] if x != join_column], random.choice(range(1, 3))) ## two columns to put in select clause
  where_combination = random.sample([x for x in columns[table1] if x != join_column], random.choice(range(1, 3))) ## two columns to put in where clause

  query = f"SELECT"
  distinct = random.choices([True, False], weights=[4.624, 95.376])[0] #proportions based on the Stack dataset statistics
  if(distinct):
    query += f" DISTINCT"

  if use_join:
      #to add one column from table1 and one column from table2 to the select clause if there is any
      query += f" {table1}.{random.choice([x for x in columns[table1] if x != join_column and x not in column_combination])}, {table2}.{random.choice([x for x in columns[table2] if x != join_column])}, "
  for column_name in column_combination:
      if use_join:
        column_name=table1+'.'+column_name
      if use_aggregate_func:
          query += f" {get_agg(column_name)}({column_name}),"
          use_aggregate_func=False
      else:
          query += f" {column_name},"

  query = query[:-1] # removing the last ','
  query += f" FROM {table1} "
  if use_join:
      query += f"{join_op} {table2} ON {table1}.{join_column} = {table2}.{join_column} "
  if use_where:
      query += "WHERE"
      for column_op in where_combination :
          op=get_op(column_op)
          column_type=coltype(column_op) # check the column type to put string values between ""
          val=random.choice(get2vals(column_op))
          if try_negation:
            negation=random.choices([True,False],weights=(8.60,91.4),k=1)[0] ##based on the Stack dataset statistics
          if op in ['IN','BETWEEN']:
            val = get2vals(column_op)
          if op =="IN":
            if negation:
              query += f" NOT({column_op} {op} {str(tuple(val))}) " ## in sql we use () not [] : but in this case we will have in ['DAM', 'MAR'] for example?! I changed the type to tuple
            else:
              query += f" {column_op} {op} {str(tuple(random.sample(val,random.randint(2,len(val)))))} "
          elif op == 'BETWEEN':
            if negation:
              query += f" NOT {column_op} {op} {val[0]} AND {random.choice(val[1:])} "
            else:
              query += f" {column_op} {op} {val[0]} AND {random.choice(val[1:])} "
          elif negation:
            query += f" NOT {column_op} {op} {val} "
          else:
            query += f" {column_op} {op} "
            if(column_type in ('s', 'b')):  # check the column type to put string values between ""
              query += f"\"{val}\" "
            else:
              query += f"{val} "

          query += random.choices(["AND"," OR","XOR"], weights=(42.02,57.96,0.02),k=1)[0]  ##based on the Stack dataset statistics
      query = query[:-4] #removing the last "AND"
  if use_group_by:
      query += f" GROUP BY {random.choice([x for x in columns[table1] if x != join_column])}"
  return query.replace('  ',' ')

def get_explanation(query,max_length=200):
  input_text = "translate Sql to English: %s </s>" % query
  features = tokenizer([input_text], return_tensors='pt').to(device)

  output = model.generate(input_ids=features['input_ids'],
               attention_mask=features['attention_mask'],max_length=max_length)

  return re.sub(r"<.*?>", "", tokenizer.decode(output[0])).strip()

##The sql to text model do not interpret well some key points, this function will enhance the results
def pre_process(query):
  if("<>" in query):
    position = query.find("<>")
    query = query[:position]+"is different than"+query[position+2:]
  if("DISTINCT" in query):
    position = query.find("DISTINCT")
    query = query[:position]+"the unique"+query[position+8:]
  while "." in query:
    dot_position = query.find(".")
    space_position = query.rfind(" ",0,dot_position)
    parenthesis_position = query.rfind("(",0,dot_position)
    query = query[:max(space_position,parenthesis_position)+1]+query[dot_position+1:] #choose the maximum of the nearest space or nearest parenthesis when dealing with agregate functoins
  return query

def get_paraphrased_sentences(model, tokenizer, sentence, num_return_sequences=5):

  torch.manual_seed(42)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)

  text =  "paraphrase: " + sentence + " </s>"

  encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
  input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

  beam_outputs = model.generate(
    input_ids=input_ids, attention_mask=attention_masks,
    do_sample=True,
    max_length=256,
    top_k=120,
    top_p=0.98,
    early_stopping=True,
    num_return_sequences=num_return_sequences
  )

  # decode the generated sentences using the tokenizer to get them back to text
  final_outputs =[]
  for beam_output in beam_outputs:
    sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
    if sent.lower() != sentence.lower() and sent not in final_outputs:
        final_outputs.append(sent)


  return final_outputs

  # decode the generated sentences using the tokenizer to get them back to text
  return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def generate_batch(dataset_size, number_of_paraphrases, output_file):
    with open(output_file, 'w', newline="") as file:
        csvwriter = csv.writer(file, delimiter=";")
        csvwriter.writerow(["question","query"])

        for i in range(dataset_size):

            group = random.choices([0,1,2,3,4], weights=(45.35,24.73,14.86,2.75,12.31),k=1)[0] # based on the Stack dataset statistics

            query = generate_query(group)
            pre_processed_query = pre_process(query)
            explanation = get_explanation(pre_processed_query)

            paraphrases = get_paraphrased_sentences(t5_model,t5_tokenizer, "paraphrase: "+explanation, number_of_paraphrases)

            for phrase in paraphrases:
                csvwriter.writerow([phrase,query])

if __name__ == '__main__':
    sys.exit(main())