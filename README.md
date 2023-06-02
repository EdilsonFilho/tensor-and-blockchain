# Project: Using tensors to identify anomalies in blockchain transactions

## Goal.
Putting into practice some concepts I learned about tensors and a healthy curiosity about Ethreum blockchian transactions. The work is just a draft, something done on a sunday afternoon, when curiosity knocks on the door of our mind. To see the results go to the code>>application.iynp folder


## Introduction
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

## Anomaly detection with Mahalanobis distances
To detect anomalies from Mahalanobis distances, it is necessary to define a threshold to consider a point as anomalous. An approach based on a reference model can be used, where a set of reference points is used to estimate the Mahalanobis distance distribution of the normal points. Points that are beyond a certain number of standard deviations from this distribution can be considered anomalous.


## What characteristics can be found when a blockchain network suffers an attack?

When a blockchain network suffers an attack, several characteristics can be observed, depending on the type of attack and the vulnerability of the network. Some of the common features that can be found are:

a) Transaction confirmation delays: If the network is being attacked with transaction spam, legitimate transactions may take longer to confirm due to competition with malicious transactions.

b) Hash rate reduction: If the network is being attacked with a 51% mining attack, the hash rate of the network can decrease as the attackers control most of the processing power of the network.

c) Chain Forks: If the network is being attacked with a double spending attack, the network nodes may perceive the existence of two different chains, since the attackers have created a transaction that spends the same balance on two different blocks. This can result in a fork in the chain, where the nodes diverge from the main chain.

d) Invalid Transactions: If the network is being attacked with an invalid transaction attack, attackers may try to send transactions that do not follow network rules, such as spending more than the available balance. Network nodes must reject these transactions, but they can cause network outages and congestion.

e) Increased Transaction Rate: If the network is being attacked with an extortion attack, attackers can try to overload the network with malicious transactions and demand a ransom to stop the attack. This can result in a significant increase in the network transaction fee.

Note: In this project we approach the analysis of gas consumption rates. We will identify the transactions that consumed the most gas. We separate between the addresses of ETH senders and receivers.


### How?
To accomplish this task, you first need to import the necessary libraries to make the API request and handle the returned JSON. Then the tensor matrix can be created from the collected data. The detection of anomalies in the "gas" column can be done by calculating the Mahalanobis distance.

Gás or gás price?

i) The "gas" refers to the amount of gas that was used in a transaction, which is the measure of computational resources required to execute an operation on the blockchain. A high amount of gas used in a transaction can indicate a complex operation or an attempt to perform a denial of service (DoS) attack on the network.

ii) "gasPrice" refers to the price of gas in units of ether (ETH), which is set by the transaction sender as an incentive for miners to process the transaction faster. A very high gasPrice rate may indicate an attempt to spam the network or to prioritize an illegitimate transaction.

Thus, depending on the type of anomaly you are looking for, it may be more relevant to look for anomalies in "gas" or "gasPrice". We used gas. 









 

