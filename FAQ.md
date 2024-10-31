# FAQ and common Problems

#### 1. Dashboard not starting (e.g. crossref error) 
Try to use 127.0.0.1 instead of localhost in address. Restart dashboard, try to use different browser (Chromium based browsers are recommended). Deactivate ad blockers and enable JavaScript. 

#### 2. Module 'ml_dtypes' has no attribute 'bfloat16' when starting dashboard
Check installation of tensorflow (version & correct installation in venv) 

#### 3. Socket Trying to (re-)establish connection 
Somehow sockets cannot make a connection to other components. Common windows problem (we recommend to use WSL). 
Check settings of protected folder access, try to restart components/computer. 


