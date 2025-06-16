import random, re, sys, argparse, torch, csv

from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModelForSeq2SeqLM



# List of columns for each table : DB SCHEMA
columns = {
        'Clients': ["client_id","first_name","last_name","year_of_birth","gender","email","city"],
        'Loans': ["loan_id","client_id","budget","duration","interest","status"],
        'Accounts': ["account_id", "client_id", "balance", "type"],
        'Deposits': ["deposit_id","account_id","amount","source"]
}
 
# Primary and Foreign keys
primary_keys = {
    "Clients": "client_id",
    "Loans": "loan_id",
    "Accounts": "account_id",
    "Deposits": "deposit_id"
}
relationships = {
    ("Loans", "client_id"): ("Clients", "client_id"),
    ("Accounts", "client_id"): ("Clients", "client_id"),
    ("Deposits", "account_id"): ("Accounts", "account_id")
}

# List of tables' names
tables = list(columns.keys())

# List of related tables (non-standalone)
related_tables = ["Clients", "Loans", "Accounts", "Deposits"]

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


# SQL-to-Text Model: takes an SQL query as an input and generates its related Natural Language Question
tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-wikiSQL-sql-to-en")
model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-wikiSQL-sql-to-en")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Paraphrasing Model: takes a natural language question and generates n paraphrases of it
t5_tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
t5_model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")


def main():
    parser = argparse.ArgumentParser(description='sql_generation')
    parser.add_argument('--dataset_size', dest='dataset_size', type=int, help='The number of queries to generate')
    parser.add_argument('--number_of_paraphrases', dest='number_of_paraphrases', type=int, help=' The number of paraphrases per query')
    parser.add_argument('--output_file', dest='output_file', type=str, help='The path to the output file')
    args = parser.parse_args()

    generate_batch(args.dataset_size, args.number_of_paraphrases, args.output_file)



# A store of possible values for each column .
#
# Parameters:
#     col (str): The column name.
#
# Returns:
#     list: The list of possible values of col.
def get2vals(col):
    first_names = [ "Madison", "Joshua", "Natalie", "Andrew", "Addison", "Ethan", "Brooklyn", "Christopher", "Savannah", "Jack", "Leah", "Ryan", "Zoey", "William", "Hannah", "Matthew", "Aaliyah", "David", "Peyton", "Mason", "Samantha", "Nathan", "Hailey", "Tyler", "Ava", "Nicholas", "Anna", "Brayden", "Kennedy", "Joseph", "Bella", "Dylan", "Sarah", "Logan", "Katherine", "Brandon", "Morgan", "Gabriel", "Taylor", "Elijah", "Natalie", "Caleb", "Gabriella", "Jackson", "Victoria", "Evan", "Claire", "Aidan", "Audrey", "Gavin", "Emily", "Liam", "Sophia", "Noah", "Olivia", "Elijah", "Emma", "Oliver", "Ava", "William", "Isabella", "James", "Mia", "Benjamin", "Charlotte", "Lucas", "Amelia", "Henry", "Harper", "Alexander", "Evelyn", "Jacob", "Abigail", "Michael", "Elizabeth", "Ethan", "Ella", "Daniel", "Lily", "Matthew", "Grace", "Aiden", "Sofia", "Jackson", "Avery", "David", "Scarlett", "Joseph", "Chloe", "Samuel", "Zoey", "Carter", "Riley", "Gabriel", "Layla", "Logan", "Audrey", "Anthony", "Skylar", "Dylan" ]
    first_names = random.choices(first_names,k=3)
    last_names = [ "Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller", "Wilson", "Moore", "Taylor", "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin", "Thompson", "Garcia", "Martinez", "Robinson", "Clark", "Rodriguez", "Lewis", "Lee", "Walker", "Hall", "Allen", "Young", "Hernandez", "King", "Wright", "Lopez", "Hill", "Scott", "Green", "Adams", "Baker", "Gonzalez", "Nelson", "Carter", "Mitchell", "Perez", "Roberts", "Turner", "Phillips", "Campbell", "Parker", "Evans", "Edwards", "Collins", "Stewart", "Sanchez", "Morris", "Rogers", "Reed", "Cook", "Morgan", "Bell", "Murphy", "Bailey", "Rivera", "Cooper", "Richardson", "Cox", "Howard", "Ward", "Torres", "Peterson", "Gray", "Ramirez", "James", "Watson", "Brooks", "Kelly", "Sanders", "Price", "Bennett", "Wood", "Barnes", "Ross", "Henderson", "Coleman", "Jenkins", "Perry", "Powell", "Long", "Patterson", "Hughes", "Flores", "Washington", "Butler", "Simmons", "Foster", "Gonzales", "Bryant", "Alexander", "Russell", "Griffin", "Diaz", "Hayes" ]
    last_names = random.choices(last_names,k=3)
    cities = [ "Springfield", "Manchester", "Jacksonville", "Cleveland", "San Antonio", "Phoenix", "Seattle", "Denver", "Atlanta", "Miami", "Dallas", "Chicago", "Houston", "Detroit", "Philadelphia", "Los Angeles", "New York City", "Boston", "San Francisco", "Las Vegas", "Orlando", "Portland", "Austin", "Nashville", "Indianapolis", "Charlotte", "San Diego", "Minneapolis", "Tampa", "Kansas City", "New Orleans", "Salt Lake City", "Baltimore", "Raleigh", "Columbus", "Pittsburgh", "St. Louis", "Milwaukee", "Buffalo", "Honolulu", "Albuquerque", "Omaha", "Louisville", "Anchorage", "Memphis", "Providence", "Hartford", "Richmond", "Oklahoma City" ]
    cities = random.choices(cities,k=3)
    emails = []
    for first_name , last_name in zip(first_names, last_names):
      email = f"{first_name.lower()}.{last_name.lower()}@cs.ma"
      emails.append(email)
    emails = random.choices(emails,k=2)

    i,j=0,0
    if col in ["loan_id","client_id","deposit_id","account_id"]:
      i,j=1,100000
    elif col in ['amount','balance','budget']:
      i,j=100,1000000
    elif col == 'year_of_birth':
      i,j=1950,2010
    elif col=='duration':
      i,j=1,30
    elif col=='interest':
      i,j=1,25
    if j!=0 and i!=0:
      fv=random.randint(i, j-3)    #fv:first value
      return [fv,random.randint(fv+2, j)]
    if col=='first_name':
      return first_names
    if col=='last_name':
      return last_names
    if col=='email':
      return emails
    if col=='gender':
      return ['Male','Female']
    if col=='city':
      return cities
    if col=='status':
      return ['current','overdue','paid off']
    if col=='type':
      return ['Deposit account','Saving account','Loan account']
    if col=='source':
      return ['Transaction','Cash','Check']
    print("ERROR Col not defined:",col)



# Checks if we can apply all the aggregation functions or just COUNT.
#
# Parameters:
#     col (str): The column name.
#
# Returns:
#     bool: False if we can apply all the aggregate function on col, True if we can apply only COUNT function on col.
def iscat(col):
  if col in ['budget','amount','balance']:
    return False
  return True



# Provides columns types.
#
# Parameters:
#     col (str): The column name.
#
# Returns:
#     str: Type of col (s for string, c for codes and ids, b for boolean, ns for numbers).
def coltype(col):
  if col in ['first_name','last_name','email','gender','city', 'status','type','source']:
    return 's'
  elif col in ["loan_id","client_id","deposit_id","account_id"]:
    return 'c'
  #if there is a boolean column add it here return 'b'
  else:
    return 'ns'


# Provides operators for each column based on its type.
#
# Parameters:
#     col (str): The column name.
#
# Returns:
#     list: List of possible operators that can be applied with col.
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


# Generates a random aggregate function for a column.
#
# Parameters:
#     col (str): The column name.
#
# Returns:
#     str: A random aggregate function for col.
def get_agg(col):
  ctype=iscat(col)
  if ctype:
    return "COUNT"
  return random.choice(aggregate_ops)


# Generate an SQL query based on a complexity group.
#
# Parameters:
#     group (int): The complexity group of the query (1,2,3,4).
#
# Returns:
#     str: An SQL query of complexity group.
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
      
      table1=random.choice(related_tables)
      candidates = []
      for (t1, c1), (t2, c2) in relationships.items():
          if t1 == table1:
              candidates.append((t2, c1, c2))
          elif t2 == table1:
              candidates.append((t1, c2, c1))

      table2, join_column1, join_column2 = random.choice(candidates)


  else:
      table1=random.choice(tables)
      join_column1 = None
      join_column2 = None

  join_op= random.choices(["JOIN","INNER JOIN","LEFT JOIN","RIGHT JOIN","FULL OUTER JOIN"], weights=(47.58,25.95,23.85,1.98,0.64),k=1)[0] ##proportions based on the STACK dataset statistics

  column_combination = random.sample([x for x in columns[table1] if x != join_column1], random.choice(range(1, 3))) ## one or two columns to put in select clause
  where_combination = random.sample([x for x in columns[table1] if x != join_column1], random.choice(range(1, 3))) ## one or two columns to put in where clause

  query = f"SELECT"
  distinct = random.choices([True, False], weights=[4.624, 95.376])[0] #proportions based on the Stack dataset statistics
  if(distinct):
    query += f" DISTINCT"

  star = random.choices([True, False], weights=[15.94, 84.06])[0] #proportions based on the Stack dataset statistics
  if(star):
    if(use_aggregate_func):
      query += f" COUNT(*)"
    else:
      query += " *"
  else:
    if use_join:
      #to add one column from table1 and one column from table2 to the select clause if there is any
      query += f" {table1}.{random.choice([x for x in columns[table1] if x != join_column1 and x not in column_combination])}, {table2}.{random.choice([x for x in columns[table2] if x != join_column2])}, "
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
      query += f"{join_op} {table2} ON {table1}.{join_column1} = {table2}.{join_column2} "
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
          
          column = column_op
          if use_join: ## add the TABLE.COLUMN in where clause in case of join
            column = f"{table1}.{column_op}"
          
          if op =="IN":
            if negation:
              query += f" NOT({column} {op} {str(tuple(val))}) "
            else:
              query += f" {column} {op} {str(tuple(random.sample(val,random.randint(2,len(val)))))} "
          elif op == 'BETWEEN':
            if negation:
              query += f" NOT {column} {op} {val[0]} AND {random.choice(val[1:])} "
            else:
              query += f" {column} {op} {val[0]} AND {random.choice(val[1:])} "
          elif negation:
            query += f" NOT {column} {op} {val} "
          else:
            query += f" {column} {op} "
            if(column_type in ('s', 'b')):  # check the column type to put string values between ""
              query += f"\"{val}\" "
            else:
              query += f"{val} "

          query += random.choices(["AND"," OR","XOR"], weights=(42.02,57.96,0.02),k=1)[0]  ##based on the Stack dataset statistics
      query = query[:-4] #removing the last "AND"
  if use_group_by:
      table=""
      if use_join:
        table=f"{table1}."
      query += f" GROUP BY {table}{random.choice([x for x in columns[table1] if x != join_column1])}"
  return query.replace('  ',' ')



# Generate a natural language question based on an SQL query.
#
# Parameters:
#     query (str): An SQL query.
#     max_length (int): The maximum number of tokens that the model is allowed to generate
#
# Returns:
#     str: The natural language question related to query.
def get_explanation(query,max_length=200):
  input_text = "translate Sql to English: %s </s>" % query
  features = tokenizer([input_text], return_tensors='pt').to(device)

  output = model.generate(input_ids=features['input_ids'],
               attention_mask=features['attention_mask'],max_length=max_length)

  return re.sub(r"<.*?>", "", tokenizer.decode(output[0])).strip()



# Preprocess the SQL query to improve its compatibility with the SQL-to-text model.
#
# Parameters:
#     query (str): The input SQL query.
#
# Returns:
#     str: The preprocessed SQL query.
def pre_process(query):
  if("<>" in query):
    position = query.find("<>")
    query = query[:position]+"is different than"+query[position+2:]
  if("DISTINCT" in query):
    position = query.find("DISTINCT")
    query = query[:position]+"the unique"+query[position+8:]
  if( "*" in query):
    query = query.replace("*","")
  while "." in query:
    dot_position = query.find(".")
    space_position = query.rfind(" ",0,dot_position)
    parenthesis_position = query.rfind("(",0,dot_position)
    query = query[:max(space_position,parenthesis_position)+1]+query[dot_position+1:] # removes the table name before each column in case of joins 
  return query


# Paraphrases a given sentence.
#
# Parameters:
#     model (object): The paraphrasing model.
#     tokenizer (object): The tokenizer associated with the model.
#     sentence (str): The natural language sentence to paraphrase.
#     num_return_sequences (int): The number of paraphrased outputs to generate.
#
# Returns:
#     list: A list containing num_return_sequences paraphrased versions of the input sentence.
def get_paraphrased_sentences(model, tokenizer, sentence, num_return_sequences=5):

  torch.manual_seed(42)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)

  text =  "paraphrase: " + sentence + " </s>"

  encoding = tokenizer.encode_plus(text,padding='max_length', return_tensors="pt")
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



# Generate a batch of NLQ-SQL pairs and save them to a CSV file.
#
# Parameters:
#     dataset_size (int): The number of NLQ-SQL examples to generate.
#     number_of_paraphrases (int): The number of paraphrased questions per SQL query.
#     output_file (str): The path to the output CSV file.
#
# Returns:
#     None. The results are written directly to the specified CSV file.
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
