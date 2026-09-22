import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import os


def clean_text(text):
    """
    Clean ticket description:
    - Convert to lowercase
    - Remove unnecessary characters
    - Remove extra spaces
    """

    text = str(text).lower()

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_and_prepare_data():

    file_path = "data/tickets.csv"

    df = pd.read_csv(file_path)

    print("\n" + "=" * 50)
    print("DATASET PREVIEW")
    print("=" * 50)

    print(df.head())

    print("\n" + "=" * 50)
    print("DATASET INFORMATION")
    print("=" * 50)

    print("Number of rows:", df.shape[0])
    print("Number of columns:", df.shape[1])

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\n" + "=" * 50)
    print("MISSING VALUES")
    print("=" * 50)

    print(df.isnull().sum())

    print("\n" + "=" * 50)
    print("DUPLICATE RECORDS")
    print("=" * 50)

    duplicate_count = df.duplicated().sum()

    print("Duplicate records:", duplicate_count)

    df = df.drop_duplicates()

    df = df.dropna(
        subset=[
            "ticket_description",
            "category"
        ]
    )

    df["clean_description"] = (
        df["ticket_description"]
        .apply(clean_text)
    )

    print("\nData preprocessing completed.")

    return df


def perform_eda(df):

    print("\n" + "=" * 50)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 50)

    print(
        "\nTotal number of tickets:",
        len(df)
    )

    print("\nTickets by category:")

    print(
        df["category"].value_counts()
    )

    print("\nTickets by priority:")

    print(
        df["priority"].value_counts()
    )

    print("\nTickets by status:")

    print(
        df["status"].value_counts()
    )

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    plt.figure(
        figsize=(10, 6)
    )

    sns.countplot(
        data=df,
        x="category",
        order=df["category"].value_counts().index
    )

    plt.title(
        "Customer Support Ticket Categories"
    )

    plt.xlabel(
        "Category"
    )

    plt.ylabel(
        "Number of Tickets"
    )

    plt.xticks(
        rotation=20
    )

    plt.tight_layout()

    plt.savefig(
        "outputs/category_distribution.png",
        dpi=300
    )

    plt.close()

    plt.figure(
        figsize=(8, 5)
    )

    sns.countplot(
        data=df,
        x="priority",
        order=["Low", "Medium", "High"]
    )

    plt.title(
        "Ticket Priority Distribution"
    )

    plt.xlabel(
        "Priority"
    )

    plt.ylabel(
        "Number of Tickets"
    )

    plt.tight_layout()

    plt.savefig(
        "outputs/priority_distribution.png",
        dpi=300
    )

    plt.close()

    print(
        "\nEDA charts saved successfully."
    )

    print(
        "\nFiles created:"
    )

    print(
        "outputs/category_distribution.png"
    )

    print(
        "outputs/priority_distribution.png"
    )


if __name__ == "__main__":

    df = load_and_prepare_data()

    perform_eda(df)