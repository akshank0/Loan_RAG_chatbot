import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, path="data/Training_Dataset.csv"):
        self.df = pd.read_csv(path)
        self._preprocess()
        self.sentences = self.df.apply(self._row_to_sentence, axis=1).tolist()
        self.sentences.append(self._generate_summary())
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = self._build_index()

    def _preprocess(self):
        df = self.df
        df['Gender'] = df['Gender'].fillna('Male').str.lower()
        df['Married'] = df['Married'].fillna('No').str.lower()
        df['Dependents'] = df['Dependents'].fillna('0')
        df['Education'] = df['Education'].fillna('Not Graduate').str.lower()
        df['Self_Employed'] = df['Self_Employed'].fillna('No').str.lower()
        df['Property_Area'] = df['Property_Area'].fillna('Urban').str.lower()

        num_cols = ['LoanAmount', 'ApplicantIncome', 'CoapplicantIncome', 'Loan_Amount_Term', 'Credit_History']
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())

        df.dropna(subset=['Loan_Status'], inplace=True)
        df['Loan_Status'] = df['Loan_Status'].astype(str).str.lower()

        self.df = df

    def _row_to_sentence(self, row):
        return (
            f"Applicant is a {row['Gender']} with {row['Education']} education, "
            f"{'married' if row['Married'] == 'yes' else 'not married'}, "
            f"{row['Dependents']} dependents, "
            f"{'self-employed' if row['Self_Employed'] == 'yes' else 'not self-employed'}. "
            f"Applicant income is ₹{row['ApplicantIncome']}, coapplicant income is ₹{row['CoapplicantIncome']}, "
            f"loan amount is ₹{row['LoanAmount']}, loan term is {row['Loan_Amount_Term']} months, "
            f"credit history: {row['Credit_History']}, property area: {row['Property_Area']}. "
            f"Loan was {'approved' if row['Loan_Status'] == 'y' else 'rejected'}."
        )

    def _generate_summary(self):
        approved = self.df[self.df['Loan_Status'] == 'y'].shape[0]
        total = len(self.df)
        avg_income = round(self.df[self.df['Loan_Status'] == 'y']['ApplicantIncome'].mean(), 2)
        return f"Out of {total} total applications, {approved} were approved. Average applicant income: ₹{avg_income}."

    def _build_index(self):
        embeddings = self.model.encode(self.sentences, show_progress_bar=True)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings))
        self.embeddings = embeddings
        return index

    def search(self, query, k=5):
        q_embed = self.model.encode([query])
        _, indices = self.index.search(np.array(q_embed), k)
        return [self.sentences[i] for i in indices[0]]
