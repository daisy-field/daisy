# Federated Intrusion Detection In Autonomous Transportation Systems

### Abstract
The increasing automation of the transportation sector and other safety-critical infrastructures
is accompanied by a major problem: Due to a steadily increasing number of cyberattacks all
over the world and the caused financial losses, image damages and failures of critical infrastructure,
companies as well as governments are forced to develop adequate countermeasures.
Intrusion Detection Systems are one of the most essential parts of cybersecurity and a highly
effective mechanism to fight the growing threat. Often it is necessary to highly automate such
systems, as the amount of security related events exceeds the capacity of human administrators.
Intrusion detection systems automatically detect security-related events, generate appropriate
alerts and are nowadays an important part of cybersecurity of complex IT systems. However,
modern intrusion detection systems have a limited effectiveness against unknown attacks and
insufficient adaptability to environments of varying complexity. In addition, the neccessary
training data is often heterogeneous, spread across multiple locations, and cannot be shared due
to regulatory, competitiveness, or privacy reasons.
Intrusion detection systems could be combined with one of the latest machine learning
concepts, called federated learning, to solve the aforementioned problems. Federated learning
enables multiple parties to construct a machine-learning model collaboratively so that the models
of different devices strengthen each other and thus achieve higher accuracy in the analysis
of safety-relevant events. In this thesis, we propose a framework using the popular federated
averaging algorithm for the aggregation of autoencoder weights to detect intrusions in modern
autonomous transportation systems. We present the customizable machine learning model,
the federated optimization algorithm, and our own communication protocol with encrypted
data transmission. In extensive experiments on the CIC-IDS17 dataset and our own collected
real-world network data, the algorithm achieves a high accuracy of around 90% in average,
a detection time of less than 0.5 seconds for anomalies, and a data throughput of around 150
samples per second. We showed that federated learning allows advancements of up to 20% in
true positive rate for individual participating clients over a non-federated training setting.

### Zusammenfassung
Die zunehmende Automatisierung des Transportsektors und anderer sicherheitskritischer Infrastrukturen
geht mit einem großen Problem einher: Aufgrund einer stetig steigenden Anzahl
von Cyberangriffen weltweit und damit einhergehende finanzielle Verluste, Imageschäden
und Ausfälle kritischer Infrastruktur sind Unternehmen sowie Regierungen dazu gezwungen,
Erkennungs- und Gegenmaßnahmen zu entwickeln und umzusetzen. Häufig ist das Ziel, solche
Systeme hochgradig zu automatisieren, da die Menge an sicherheitsrelevanten Ereignissen die
Kapazität von menschlichen Administratoren übersteigt. Intrusion Detection Systeme erkennen
automatisch sicherheitsrelevante Ereignisse, generieren entsprechende Alarmmeldungen und
sind heutzutage ein wichtiger Bestandteil der Cybersicherheit komplexer IT-Systeme. Moderne
Intrusion Detection Systeme haben allerdings nur eine begrenzte Wirksamkeit gegen unbekannte
Angriffe und eine unzureichende Anpassungsfähigkeit an unterschiedlich komplexe
Umgebungen. Außerdem mangelt es häufig an Daten zum Trainieren komplexer Intrusion Detection
Systeme, da diese meistens auf verschiedenste Geräte verteilt anfallen und nicht ohne
weiteres zusammengeführt werden können.
Intrusion Detection Systeme könnten mit einem der neuesten Konzepte des maschinellen
Lernens, dem sogenannten Federated Learning, kombiniert werden, um die genannten Probleme
zu lösen und um eine wesentlich verbesserte automatische Erkennung sicherheitsrelevanter
Ereignisse zu erzielen. Durch den Austausch von Modelldaten stärken sich im Federated
Learning die Modelle verschiedener Geräte untereinander und erreichen damit eine
höhere Genauigkeit in der Analyse von sicherheitsrelevanten Ereignissen. In der vorliegenden
Arbeit werden verschiedene Ansätze von Federated Learning für die Nutzung der Angriffserkennung
untersucht. Es wird ein Federated Learning Framework implementiert, das
die zuverlässige Erkennung von Cyberattacken in Netzwerken erlaubt, trotzdem vollständig
flexibel auf unterschiedliche Anwendungsgebiete zugeschnitten werden kann und damit viel
Potenzial für zukünftige Weiterentwicklungen bietet. Anschließend wird für die Evaluation
des Frameworks ein Autoencoder verwendet, um im Kontext eines autonomen Transportsystems
den Federated Learning Ansatz zu evaluieren. In umfangreichen Experimenten mit dem
CIC-IDS17-Datensatz und mit gesammelten realistischen Netzwerkdaten erreicht unser Algorithmus
eine hohe Genauigkeit von 90% und eine Erkennungszeit von durchschnittlich weniger
als 0,5 Sekunden für Angriffe. Außerdem haben wir gezeigt, dass Federated Learning eine
Verbesserung der True-Positive-Rate von bis zu 20% für einzelne Geräte gegenüber einer klassischen
Trainingsumgebung ermöglicht.

### License
This project is licensed under the terms of the MIT license.
