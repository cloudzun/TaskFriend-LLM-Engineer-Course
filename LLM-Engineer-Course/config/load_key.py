def load_key(confirmation=True):
    import os
    import json
    import dashscope
    import getpass

    # Helper to detect Jupyter environment
    def in_notebook():
        try:
            from IPython import get_ipython
            if 'IPKernelApp' in get_ipython().config:  # Jupyter notebook
                return True
            if 'VSCODE_PID' in os.environ:  # VS Code notebook
                return True
        except:
            pass
        return False

    file_name = '../Key.json'

    if os.path.exists(file_name):
        with open(file_name, 'r') as file:
            Key = json.load(file)
        existing_key = Key["DASHSCOPE_API_KEY"].strip()
        os.environ['DASHSCOPE_API_KEY'] = existing_key

        print(f"Detected existing API key: {existing_key[:4]}...{existing_key[-4:]}")
        
        # Use the confirmation argument to skip the prompt
        if confirmation:
            use_existing = input("Continue using this API key? (y/n): ").strip().lower()
        else:
            print("Auto-confirmation enabled. Using existing API key.")
            use_existing = 'y'  # Automatically proceed

        if use_existing != 'y':
            new_key = getpass.getpass("Input new API key: ").strip()
            Key = {"DASHSCOPE_API_KEY": new_key}
            with open(file_name, 'w') as json_file:
                json.dump(Key, json_file, indent=4)
            os.environ['DASHSCOPE_API_KEY'] = new_key
    else:
        api_key = getpass.getpass("No key.json found. Input your API key: ").strip()
        Key = {"DASHSCOPE_API_KEY": api_key}
        with open(file_name, 'w') as json_file:
            json.dump(Key, json_file, indent=4)
        os.environ['DASHSCOPE_API_KEY'] = api_key

    dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]


if __name__ == '__main__':
    load_key()
    import os
    print("The configured API key is:", os.environ['DASHSCOPE_API_KEY'][:4] + "..." + os.environ['DASHSCOPE_API_KEY'][-4:])