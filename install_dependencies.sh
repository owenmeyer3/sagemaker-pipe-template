# ./install_dependencies
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
sudo yum install unzip -y
unzip awscliv2.zip
sudo ./aws/install
pip3 install --upgrade pip
pip3 install sagemaker==3.13.1 --no-deps sagemaker-core sagemaker-train sagemaker-serve sagemaker-mlops
pip3 install -r requirements.txt  -v