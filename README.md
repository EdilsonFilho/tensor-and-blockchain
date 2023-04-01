# Project: Using tensors to identify anomalies in blockchain transactions

#### Author: edilsonfilho@lesc.ufc.br

The use of tensors can help detect anomalies in blockchain transactions due to their ability to model complex patterns and relationships between data.
Tensors are multidimensional mathematical data structures that can be used to represent data with multiple dimensions, such as images, videos, and time series. In a blockchain transaction, tensors can be used to represent various dimensions of data such as time, transaction types, amounts transferred, and addresses involved.
For example, a tensor can be created to represent all transactions that occur in a certain period of time on a specific blockchain. Different dimensions of the tensor can represent different aspects of transactions, such as time of day, source and destination addresses, and amount transferred.
With this data modeled in tensors, machine learning algorithms can be applied to detect transaction anomalies. For example, anomaly detection algorithms can be trained to identify unusual patterns in transactions, such as large amounts of money transferred to unknown addresses or transactions that occur at unusual times.
In this way, the use of tensors can help identify suspicious transactions in blockchains and provide users with greater security and transparency in their transactions.

##### Development environment setup
In terminal 
1- Start

```sh
pip install virtualenv
```

```sh
virtualenv tensor-blockchain
```
```sh 
source tensor-blockchain/bin/activate
```

2- Install dependencies
```sh
pip3 install scipy
```
```sh
python3 -m pip install matplotlib
```






 

