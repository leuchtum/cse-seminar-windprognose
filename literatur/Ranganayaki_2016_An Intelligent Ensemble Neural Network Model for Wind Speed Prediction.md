# Alis Zusammenfassung
## 1 Introduction

TODO: Unterteilen in Themenbereiche. Das ist zu viel Input auf einmal :D

- Wind ist natürliche Ressource, die massenweise zur Verfügung steht
- Windenergie ist dazu noch sauber (keine Umweltverschmutzung) und kann auch so erzeugt werden, dass es den regulären Bedarf des Landes deckt
- Wind kann über Richtung, Geschwindigkeit und zeitliches Aufkommen eingeordnet werden
- Gewinnen der Windenergie basiert auf Windkraft bzw. Windgeschwindigkeit
- Windstärke/-geschwindigkeit i.d.R. nichtlinear und von Natur aus schwankend
- Windvorhersage zur Steigerung der erzeugten Energie
- Windgeschwindigkeitsvorhersage hilft Ausgleich zwischen erforderlichem Bedarf und erzeugter Energie zu finden
- Modell zur Vorhersage (Ideal, also mit hoher Genauigkeit und Zuverlässigkeit) dient als effektives Werkzeug zur Optimierung der Betriebskosten und zur Verbesserung der Betriebseigenschaften des Netzsystems
- Diese Arbeit trägt zur Entwicklung von NN für die Durchführung einer effektiven Windstärkenvorhersage bei
- Weitere Faktoren, die die Windgeschwindigkeit beeinflussen: Luftfeuchtigkeit, Feuchtigkeit in der atmosphärischen Luft, atmosphärischer Druck, Temperatur, Niederschläge
- künstliche neuronale Netz ist intelligente Computertechnik, die der Charakteristik des menschlichen biologischen neuronalen Netzes ähnelt
- Hauptmerkmale NN: Nichlinearität, Anpassungsfähigkeit, Fähigkeit große Datenmengen zu verarbeiten
- Wegen der genannten Eigenschaften ist NN effektives Werkzeug zur Vorhersage von Windgeschwindigkeit auf Grundlage der definierten Inputs
- Leistungskennzahl ("performance metric") ist der mittlere quadratische Fehlerwert, der zur Messung der Qualität der prognostizierten Windgeschwindigkeit mit Hilfe des neuronalen Netzes verwendet wird
- Hauptproblem bei anderen Papern: Fixierung der versteckten Neuronen im hidden Layer (hidden Neurons und hidden Layer wichtig für Berechnung minimaler Fehler)
- In dieser Arbeit wird das entwickelte Ensemble-Neuronalnetz-Modell mit den vorgeschlagenen 102 Kriterien getestet, um die geeignete Anzahl von versteckten Neuronen im hidden layer jedes MLP, Madaline, BPN und PNN festzulegen.
- Kriterien werden aus Übereinstimmung mit Konvergenzkriterium ausgewählt und vorgeschlagenes NN wird in dieser Arbeit an Windgeschw.prognose angepasst
- Hauptaugenmerk: Minimaler Fehler, Verbessung der Netzstabilität, bessere Genauigkeit im Vergleich zu anderen existierenden Ansätzen

## 2 Related Work

- Hauptziel: bestimmte ensemble NN Modelle zu entwicklen und die Anzahl der versteckten Neuronen im hidden Layer zu entwerfen und Modell für eine Windgeschw.vorhersage anzuwenden, basierend auf detaillierten Analysen, die in früheren Arbeiten zu diesem Thema durchgeführt wurden
- In der Literatur gibt es schon viele Ansätze zu dem Thema:
    * Windgeschwindigkeitsvorhersage: physikalische Ansätze, Zeitreihen, statistische Methoden und auch Ansätze des maschinellen Lernens.
    * Windenergieerzeugung: BPN (Fehlerminimierung und genauere Vorhersagen)
    * Aufzählen von ca. 26 Papern und deren verwendeten Modellen/Algorithmen
- Rückschlüsse aus review der in den Papern verwendeten NN zur Windgeschwindigkeitsvorhersage:
    * Zufällig gewählte Anzahl versteckter Neuronen führte zu Over-/Underfitting
    * Fehler wurde nicht erheblich verringert
    * Bestehende Methoden gehen mit Versuchen vor, dabei ist die Anzahl der versteckten Neuronen nicht festgesetzt
- Aus diesen Gründen wird in dieser Arbeit ein NN ensemble-Modell entwickelt, das die Eigenschaften von MLP, BPN (back propagation network), Madaline und PNN (proposed probabilistic neural network) kombiniert, um die Windgeschw. vorherzusagen

## 3 Problem Formulation

- Windenergie erzeugt in Windpark hängt von stochastischer Natur der Windgeschwindigkeit ab und unerwartete Abwichung der Windkraftleistung führt zu Erhöhung der Betriebskosten
- Zusammenhang Windgeschwindigkeit und Windleistung sind hoch-nichtlinear --> Fehler in Windgeschwindigkeitsvorhersage führt zu großem Fehler in Windenergieerzeugung
- NN lernt aus Daten der Vergangenheit und sagt mit dessen Hilfe zukünftige Daten voraus, ein genaues Modell ermöglicht wirtschaftlichen Betrieb um Anforderungen der Stromkunden zu erfüllen
- Windgeschwindigkeitsverhalten nicht-stationär, was auf dynamische Eigenschaft während versch. Perioden deutet, die zu Schwankungen von Input und Output führt
- Ziel: geeignetes Kriterium zur Bestimmung der Anzahl der versteckten Neuronen, um Windgeschw. mit höherer Genauigkeit und minimalem Fehler vorherzusagen
- Zu wenig versteckte Neuronen: lokales Minimum / zu viele: Instabiles Modell
- Performance wird gemessen an minimal Mean Square Error (MSE)
    * MSE = sum_{i=1}^{N}(frac{(Y_{predict}-Y_{actual})^2}{N})
    * Y_{predict} ist vorhergesagter Output
    * Y_{actual} ist der wahre Output
    * N ist Anzahl der Sample

## 4 Modeling the Proposed Ensemble Neural Network Architecture

- Es existieren eigentlich noch keine Methoden, um die Anzahl der hidden Neurons zu wählen
- Im paper werden alle neuen Kriterien ausprobiert, die das Konvergenztheorem erfüllen und letztlich das ausgewählt, das Fehler im Trainingsprozess optimal reduziert

### 4.1 Design of the Proposed Ensemble Neural Network

- Ensemble-NN:
    *  Ausgänge der unabhängig voneinander trainierten NN werden verglichen
    * Beinhalten Multilayer Perceptron (MLP), Multilayer Adaptive Linear Neuron (Madaline), Back Propagation Neural Network (BPN), Probabilistic Neural Network (PNN)
    * Jeder Output wird gemittelt, um einen besseren Output in Abhängigkeit des Problems zu erhalten

#### Module 1 (MLP)

- Wird zusammen mit überwachtem Lernansatz verwendet
- nichtlineare Sigmoidfunktion als Aktivierungsfunktion
- Anzahl der hidden Neuronen werden durch Kriterium festgelegt
- Jedes Layer des MLP ist innerhalb der hidden Neurons durch synaptische Gewichte verbunden

#### Module 2 (Madaline)

- Einzelne Adalines kombiniert, sodass Output einiger Input anderer wird -> Netz wird so zu Madaline
- Vorgehen:
    * Initialisierung der Gewichte zwischen Input und hidden units (kleine positive Zufallswerte)
    * Input wird präsentiert und anhand der gewichteten Verschaltungen wird Netzeigabe für die hidden units berechnet
    * Aktivierung wird angewendet um Ausgäge der hidden units zu erhalten
    * hidden Output ist Input für output layer
    * passende gewichtete Verbindungen lassen netto in- und output berechnen
    * Vergleich mit target und Gewichtsanpassungen für neuen Schritt

#### Module 3 (BPN) 

- Input layer ist verbunden mit hidden layer ist verbunden mit output layer mit gewichteten Verbindungen
- Während backpropagation des Trainings werden Signale in umgekehrter Reihenfolge gesendet
- Erhöhen der Anzahl der hidden layers führt zu Rechenkomplexität des Netzes. Es wird grundsätzlich nur ein hidden layer verwendet
- Kriterium ist in Trainingsalgorithmus eingebaut, um Anzahl der hidden Neurons im single hidden layer zu reparieren
- bias (ist das in diesem Zusammenhang der Fehler?) wird dem hidden layer und dem output layer zur Verfügung gestellt, um Einfluss auf den zu berechnenden netto input zu haben

#### Module 4 (PNN)

- Konzept: Bayesian Classification und Schätzung von Wahrscheinlichkeitsdichtefunktionen
- Inputvektoren werden Bayes-optimiert klassifiziert (2 Klassen)
- f_{A}(x):
    * dient als Schätzer, solange die übergeordnete Dichte glatt und kontinuierlich ist
    * Wächst die Anzahl der Datenpunkte, nähert sich f_{A}(x) der übergeordneten Dichtefunktion an
    * Ist die Summe der Gaussverteilungen

#### Als Ensemble-NN

- Alle vier Module beginnen ihren Trainingsprozess bis Fehler vernachlässigbar
- Durchschnittwert der vier Module ergibt endgültigen Output

### 4.2 The proposed Training Algorithm of Ensemble Neural Network
- Step 1: Initialisieren der notwendigen Parameter aller vier Module
- Step 2: Kriterium zur Festlegung der Anzahl der hidden Neurons in alle Module einführen
- Step 3: Übergabe Input- und Zielvektorpaar an die Module. Jeweils Trainingsdaten und Testdaten.
- Step 4: Berechne den netto Input der einzelnen Module und erhalte die entsprechenden Outputs indem Aktivierung auf die berechneten Netz-Inputs angewendet wird (Outputs: Y_{MLP}, ... für jedes Modul H_{MLP}, ...)
- Step 5: Ensemble-Netz als Aggregation der Module entwickeln und endgültige Windgeschwindigkeit ermitteln
    * Formel 2 im Paper (Aufstellen der Formeln des Ensemble-Netzes)
- Step 6: Jedes Modul trainieren und Fehler berechnen (Formel 3)
- Step 7: Auswahl Anzahl der hidden Neurons basierend auf Fehler
- Step 8: Ausgewähltes Kriterium und minimaler Fehlerwert ausgeben
- Step 9: Test auf Abbruchbedingungen (Erreichen eines minimalen Fehlerwerts oder bestimmte Anzahl Epochen)

## 5 Numerical Experimentation ans Simulation Results
- Arbeit konzentriert sich darauf Anzahl der hidden Neurons festzulegen und damit MSE zu reduzieren und genaue Windgechwindigkeit vorherzusagen
- Festlegung hidden Neurons sollte zu höherer Genauigkeit und schnellerer Konvergenz führen
- 102 mögliche Kriterien, um hidden Neurons basierend auf Fehler während Lernphase zu berechnen
- Windgeschwindigkeit wird aus output des Ensemble NN berechnet nach der "denormalization procedure"

### 5.1 Description of Real Time Wind Farm Datasets
- Viel Input woher die Daten, in welchem Zeitraum, wie gemessen, ...
- 90000 Datensample über 2 Jahre (04.2013-05.2015), entspricht 5 Sample pro Stunde
- Windrichtung:
    * gemessen mit wind vane
    * Vane zeigt in Richtung von der der Wind kommt
    * Richtungsmessung startet bei true North in grad
- Windgeschwindigkeit:
    * Anemometer
- Temperatur:
    * Thermometer

### 5.2 The Proposed Wind Speed Prediction Ensemble NN Simulation
- Jedes Netz mit in 5.1 erwähnten Daten trainiert, bis minimaler Fehler erreicht wird
- Input und Output Variablen, die Trainingsprozess als Input/Output Neuronen dienen sind in Tabelle 3
- Erster Schritt ist Daten normalisieren mit min-max-Approach:
    * Formel 4: Xnorm = ((Xactual-Xmininput) / (Xmaxinput-Xmininput)) * (Xmaxtarget-Xmintarget) + Xmintarget
    * Xactual, Xmininput, Xmaxinput = Echter Input und min/max des Inputs
    * Xmaxtarget, Xmintarget = max/min der Zieldaten der Windfarmdaten
- Nettoinput ist gewichtete Summe der Inputs und Akrtivierungsfunktion der Ensembles wird verwendet, um Output zu berechnen
- Tabelle 5 zeigt die 102 Kriterien, die die hidden Neuronen im Ensemlbe Network festlegen. Konvergenztheorem im Appendix
    * Jedes Kriterium wird auf die einzelnen Ensembles angewendet
    * Mean Square Error (MSE) wird für die einzelnen Ensembles berechnet

- Letztlich wird jedes Kriterium auf alle Netze angewendet und im Anschluss der MSE von allen gemittelt berechnet. Das Kriterium, bei dem der kleinste gemittelte MSE Wert herauskommt wird verwendet um die Netze zu trainieren und um die anzahl der hidden Neurons des Ensemble Netzes zu bestimmen.
- Gewähltes Kriterium:
    * (6n + 7)/(n - 3)
    * MSE: 0.015150

## 6. Discussion on the Simulated Results

# Daniel Zusammenfassung

## 1. Introduction

### Wind

- Wind wird beschrieben durch:
  - Richtung
  - Geschwindigkeit
  - Zeitpunkt des Auftretens
- Wind verhält sich im Allgemeinen nichtlinear und fluktuiert stark
- Windprognose mit hoher Genauigkeit und Verlässlichkeit stellen ein effizientes Werkzeug zur Optimierung der Betriebskosten dar.
- Weitere Faktoren, die die Windgeschwindigkeit beeinflussen:
  - Luftfeuchte
  - Druck
  - Temperatur
  - Niederschlag

### Neural Networks

- Hauptvorteil von NNs:
  - Nichtlinearität
  - Anpassbarkeit
  - Fähigkeit große Datenmengen zu verarbeiten
  - Fähigkeit der Generalisierung
- Zur Messung der Prognosequalität wurde MSE genutzt
- Hauptfokus liegt auf:
  - Stabilität
  - Bessere Genauigkeit im Vergleich zur existierenden Forschung
- Anzahl der Neuronen nur schwer festzusetzen, daher Entwicklung von 102 Kriterien um die Anzahl der Neuronen in den einzelnen SubNNs festzusetzen.

## 2. Related Work

- 25+ verschiedene Paper werden erwähnt, allerdings wird auf fast keins tiefer eingegangen.
- Die Forscher haben folgende Probleme bei den vorhandenen Paper'n festgestellt:
  - Die Anzahl der Neuronen in den jeweiligen neuronalen Netzen wurden teils zufällig gewählt; Es kommt zu Over/Underfitting
  - Die Prognosegüte wurde nicht signifikant verbessert
  - Die Anzahl der Neuronen werden teilweise nicht fixiert

## 3. Problem Formulation

- Die erzeugte Energie eines Windparks ist stark abhängig von dem stochastischen Verhalten des Windes
- Die Beziehung zwischen Windgeschwindigkeit und der erzeugten Energie weißt eine hohe Nichtlinearität auf
- Die genaue und verlässliche Windprognose ist Vorraussetzung für eine gute Gesamtnetzführung und fortschrittlichen Regelstrategien

## 4. Modeling the Proposed Ensemble Neural Network Architecture

### 4.1 Design of the Proposed Ensemble Neural Network

- Definition Ensemble Network: Kombiniert den Output selbstständig arbeitender Netzwerke
- Hier vier selbstständige Netze:
  - MLP Multilayer Perceptron
  - Madaline: Miltilayer Adaptive linear neuron
  - BPN: Back Propagation neural network
  - PNN: probabilistic neural network
- Jedes dieser Netzwerke wird (unter Berücksichtigung des gewählten Kriteriums) mittels des jeweiligen Trainingsalgorithmus trainiert.
- **Zeile 11, Seite 4: Der Output der einzelnen Netzwerke wird gemittelt!**
- Hierdurch wird die Fähigkeit der Generalisierung und die Stabilität verbessert. (Adressiert das "lokale Minimum"-Problem und die verzögerte Konvergenz)
- Figure 2: Die einzelnen neuronalen Netze führen die Berechnungen mit den Input Daten unabhängig von einander aus. Die berechneten Ergebnisse werden in der nachfolgenden Schicht weiter verarbeitet und das finale Ergebnis berechnet.
- Der Mittelwert der Subnetze ergibt das Ergebnis

#### Module 1: MLP

- supervised learning
- nonlinear activation function: logistic sigmoid function

> **1. WIDERSPRUCH**: In Module 1 auf Seite 4 heißt es, Sigmoid wird als Aktivierungsfunktion genutzt. In Tabelle 4 auf Seite 8 heißt es, "Binary linear activation funtion" wird genutzt.

> **2. WIDERSPRUCH**: Eine Aktivierungsfunktion "Binary linear activation funtion" macht für mich keinen Sinn. Entweder Binär, oder Linear!

> **3. WIDERSPRUCH**: Gleiches gilt für BPN: Tabelle 4 auf Seite 8 heißt es, "Binary sigmodial funtion" wird genutzt. Entweder Binär, oder Simodial!

#### Module 2: Madaline

- Einfache ADALINEs zusammengefügt -> M(any) ADALINE
- Initialisierung der Gewichte zwischen den ADALINEs: Random numbers

TODO Noch mehr dazu schreiben.

#### Module 3: BPN

- Nur eine Hidden Layer
- Bias wird für Hidden und Outputneuronen genutzt

#### Module 4: PNN

- Basiert auf Bayesian Klassifizierung
- Prognostiziert über Dichtefunktionen

> **4. WIDERSPRUCH**: Meines Wissens nach werden PNNs für Klassifizierungsprobleme und nicht Regressionsprobleme genutzt

### 4.2 The proposed training algorithm of Ensemble Neural Network

1. Parameter in den Subnetzen initialisiere
2. Anzahl der hidden Neuronen anhand des Kriteriums festlegen
3. Trainingsdaten werden präsentiert
4. 