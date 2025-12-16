wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"
wrk.body = [[
    {
       "model": "./model/qwen2_5-1_5b-instruct",
       "messages": [
           {"role": "system", "content": "You are a helpful AI assistant."},
           {"role": "user", "content": "Say hi!"}
       ]
   }
]]