import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Access the variables
api_key = os.getenv("API_KEY")
db_user = os.getenv("DB_USERNAME")
db_password = os.getenv("DB_PASSWORD")

# Print for debugging (remove in production)
print(f"API Key: {api_key}")
print(f"DB User: {db_user}")
print(f"DB Password: {db_password}")

print ("hello world")