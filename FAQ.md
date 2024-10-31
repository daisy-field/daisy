# FAQ and common Problems

#### 1. Dashboard not starting (e.g. crossref error) 
Try to use 127.0.0.1 instead of localhost in address. Restart dashboard, try to use different browser (Chromium based browsers are recommended). Deactivate ad blockers and enable JavaScript. 

#### 2. Module 'ml_dtypes' has no attribute 'bfloat16' when starting dashboard
Check installation of tensorflow (version & correct installation in venv) 

#### 3. Socket Trying to (re-)establish connection 
Somehow sockets cannot make a connection to other components. Common windows problem (we recommend to use WSL). 
Check settings of protected folder access, try to restart components/computer. 

#### 4. PCAP files aren't read
For network traffic, PyShark is used. This is a library using tshark in the background. This means that it is dependent on the tshark installation. 
It may be required to execute the code with root/admin permissions, as tshark might be configured to deny non-root users to use its features.
On Windows machines it was observed, that pyshark has trouble using tshark, despite correct installation and path variables. WSL or Linux might be required in these cases.

#### 5. Live Network traffic isn't captured
Refer to question 4, as the Live Network capture uses PyShark and suffers from the same problems.
