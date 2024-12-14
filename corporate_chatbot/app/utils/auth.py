def check_credentials(username: str, password: str) -> bool:
    try:
        with open('credentials.txt', 'r') as f:
            stored_username = f.readline().strip()
            stored_password = f.readline().strip()
            return username == stored_username and password == stored_password
    except Exception as e:
        print(f"Error reading credentials: {e}")
        return False