# Alis Zusammenfassung
## Introduction

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
- Wegen der genannten Eigenschaften ist NN effektives Werkzeug zur Vorhersage von Windgeschwindigkeit auf Grundlage der definierten Eingabeparameter
- Leistungskennzahl ("performance metric") ist der mittlere quadratische Fehlerwert, der zur Messung der Qualität der prognostizierten Windgeschwindigkeit mit Hilfe des neuronalen Netzes verwendet wird
- Hauptproblem bei anderen Papern: Fixierung der versteckten Neuronen im hidden Layer (hidden Neurons und hidden Layer wichtig für Berechnung minimaler Fehler)
- In dieser Arbeit wird das entwickelte Ensemble-Neuronalnetz-Modell mit den vorgeschlagenen 102 Kriterien getestet, um die geeignete Anzahl von versteckten Neuronen im hidden layer jedes MLP, Madaline, BPN und PNN festzulegen.
- Kriterien werden aus Übereinstimmung mit Konvergenzkriterium ausgewählt und vorgeschlagenes NN wird in dieser Arbeit an Windgeschw.prognose angepasst
- Hauptaugenmerk: Minimaler Fehler, Verbessung der Netzstabilität, bessere Genauigkeit im Vergleich zu anderen existierenden Ansätzen

## Related Work

- Hauptziel: bestimmte ensemble N Modelle zu entwicklen und die Anzahl der versteckten Neuronen im hidden Layer zu entwerfen und Modell für eine Windgeschw.vorhersage anzuwenden, basierend auf detaillierten Analysen, die in früheren Arbeiten zu diesem Thema durchgeführt wurden
- In der Literatur gibt es schon viele Ansätze zu dem Thema:
* Windgeschwindigkeitsvorhersage: physikalische Ansätze, Zeitreihen, statistische Methoden und auch Ansätze des maschinellen Lernens.
* Windenergieerzeugung: BPN (Fehlerminimierung und genauere Vorhersagen)


# Daniel Zusammenfassung

## Introduction

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

## Related Work

- 25+ verschiedene Paper werden erwähnt, allerdings wird auf fast keins tiefer eingegangen.
- Die Forscher haben folgende Probleme bei den vorhandenen Paper'n festgestellt:
  - Die Anzahl der Neuronen in den jeweiligen neuronalen Netzen wurden teils zufällig gewählt; Es kommt zu Over/Underfitting
  - Die Prognosegüte wurde nicht signifikant verbessert
  - Die Anzahl der Neuronen werden teilweise nicht fixiert

## Problem Formulation

- Die erzeugte Energie eines Windparks ist stark abhängig von dem stochastischen Verhalten des Windes
- Die Beziehung zwischen Windgeschwindigkeit und der erzeugten Energie weißt eine hohe Nichtlinearität auf
- Die genaue und verlässliche Windprognose ist Vorraussetzung für eine gute Gesamtnetzführung und fortschrittlichen Regelstrategien