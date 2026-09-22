import pandas as pd
import random

random.seed(42)

ticket_templates = {
    "Login Issue": [
        "I cannot login to my account",
        "My password reset is not working",
        "I forgot my password",
        "I am unable to sign in",
        "Login page is rejecting my password",
        "My account login is not working",
        "I cannot access my account",
        "The system says my password is incorrect",
        "I am locked out of my account",
        "Please help me recover my login",
    ],

    "Application Error": [
        "The application shows an error while saving",
        "Application crashes when I upload a file",
        "I am getting an error message",
        "The application stopped working",
        "There is an error when opening the application",
        "The system displays an unexpected error",
        "I cannot save my information because of an error",
        "The application crashes after submitting the form",
        "An error appears whenever I try to update data",
        "The application is showing an internal error",
    ],

    "Report": [
        "I need the monthly sales report",
        "Please generate the annual sales report",
        "I need a report for last month",
        "How can I download the sales report",
        "Please provide the customer report",
        "I need the weekly performance report",
        "Generate the financial report",
        "I want to download the monthly report",
        "Please send me the transaction report",
        "I need a report of all completed orders",
    ],

    "Account Update": [
        "I need to change my registered mobile number",
        "Please update my email address",
        "I want to change my account details",
        "Please update my phone number",
        "I need to change my registered address",
        "How can I update my profile information",
        "Please change my account email",
        "I want to update my personal details",
        "My contact information needs to be changed",
        "Please update my account information",
    ],

    "Performance": [
        "The application is extremely slow today",
        "The system takes too long to load",
        "The application response is very slow",
        "Pages are loading slowly",
        "The system performance has become very poor",
        "The application is taking too long to respond",
        "The website is very slow",
        "The dashboard takes several minutes to open",
        "The system is responding slowly",
        "Application performance is very bad today",
    ],
}

priorities = ["Low", "Medium", "High"]
statuses = ["Open", "In Progress", "Closed"]

rows = []

ticket_number = 1

for category, templates in ticket_templates.items():

    for _ in range(45):

        description = random.choice(templates)

        rows.append({
            "ticket_id": f"T{ticket_number:03d}",
            "ticket_description": description,
            "category": category,
            "priority": random.choice(priorities),
            "status": random.choice(statuses)
        })

        ticket_number += 1


df = pd.DataFrame(rows)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv("data/tickets.csv", index=False)

print("Dataset created successfully!")
print("Total records:", len(df))
print("\nCategory distribution:")
print(df["category"].value_counts())