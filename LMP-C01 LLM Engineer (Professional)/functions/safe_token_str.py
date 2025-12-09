def safe_token_str(token):
    """Convert token to string safely (handles bytes/str)"""
    return token.decode('utf-8') if isinstance(token, bytes) else token