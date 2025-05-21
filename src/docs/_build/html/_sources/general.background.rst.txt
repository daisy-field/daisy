Background
===========

Cybersecurity in modern networks presents significant challenges arising from the heterogeneity
of devices, dynamic topology changes, varying computational capabilities, and limited network bandwidth.
Highly automated intrusion detection systems (IDS) with integrated machine learning methods
became one of the most important defence methods against potential cybersecurity threats,
by continuously monitoring network traffic and reporting potential security violations automatically.
However, training such machine learning models for the automatic detection of threats is still a challenging
task, as the data is heterogeneous, spread across different locations and often non-iid.

Federated Learning
-------------------

In 2015, Google introduced federated learning (FL) as a solution to address the challenges
related to distributed datasets and insufficient data availability. The goal was to enhance
the prediction accuracy of Google’s keyboard next-word suggestions by learning from input
behaviours of all smartphone users while maintaining their privacy rights. In contrast to
traditional machine learning, where all the data is collected on a central server ,



.. image:: /images/traditional.png
  :width: 40%
  :alt: Traditional Machine Learning

.. image:: /images/federated_learning.png
  :width: 40%
  :alt: Traditional Machine Learning


FL algorithms construct machine learning models using models learned on distributed datasets,
located on separate nodes.
Typically, the process starts with training a model on the nodes
with the locally available data. The trained models are then transmitted to a central aggregation
server while maintaining data privacy.


The central server generates a global model by combining the received node models using an aggregation method. FL not only enhances the speed and efficiency of real-time data processing, but also increases security
and privacy by keeping sensitive data closer to its point of origin, thereby reducing the risks of
breaches compared to transmitting large volumes of data over potentially insecure networks. It
also offers significant cost advantages and lower latency, enabling faster responses to security
threats and improving the overall performance of IDS.


Personalized Federated Learning
--------------------------------
(coming soon)


Decentralized Federated Learning
---------------------------------
(coming soon)


Model Poisoning
--------------------------------
(coming soon)
