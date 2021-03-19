# 01_literature_review

Dieses Dokument stellt sieben Paper vor, die in Bezug auf Windprognose interessant sind.

### Absatz 1

> **ACHTUNG**: [**SIEHE DOKUMENT**](./Orpia_2014_Time%20Series%20Analysis%20using%20Vector%20Auto%20Regressive%20(VAR)%20Model%20of%20Wind%20Speeds.pdf)

* Bei Windprognose ist es möglich, örtliche und zeitliche Abhängigkeiten mit zu berücksichtigen
* Wenn nur Daten von einem Platz vorliegen: -> Model via:
  + AR
  + VAR
  + ARIMA

### Absatz 2

* Vorhandenes Paper: Prognose von Windgeschw. über tägliche Daten
* Genutzte Variablen:
  + Windgeschw.
  + Luftfeuchte
  + Temp.
  + Druck
* Diese Faktoren beeinflussen die Windturbine
* Diese Faktoren sind stationär (nach dem Augmented Dickey-Fuller-Test)
* Ergebnisse von vorhandenen Paper:
  + Windgeschw. erklärbar mit AR
  + Steigt die Temperatur, ? Luftfeuchte? oder der Druck, singt Windgeschw.
  + Gezeigt durch Impulse Response Funktion

### Absatz 3

* Vorhandenes Paper: Prognose von Windgeschw. für Windkraftanlagen mittels Sparse-VAR (sVAR)
* Horizont: 5min
* Es werden nur Lastdaten der Turbinen zur Prognose genutzt
* Vergleich mit konventionallen VAR und AR
* Eröffnet Verbesserungsmöglichkeiten (Zuverlässigkeit etc.)

### Absatz 4

* Vorhandenes Paper: Interessant, analysiert Windgeschwindigkeiten in verschiedenen Höhen in Abhängigkeit der Höhe.
* Nutzt VAR Model

### Absatz 5

* Vorhandenes Paper: Kurzzeitige Windprognose innerhalb eines Windparks über die aktuelle Leistung der anderen Turbinen
* Nutzt VAR Model
* Umgehen Gefahr von Overfitting über eine clevere Coefficient Matrix, die Windrichtung, Geschw. und Leistung berücksichtigt.
* Hohes Potential zur Verbesserung von Lastprognose/Modellierung

### Absatz 6

* Vorhandenes Paper: Windprognose an verschiedenen Orten
* Nutzt spezielles VARTA Model (time-dependent intercept/ -> ?"zeitabhängige Interpolation"?)
* Lang und Kurzzeitprognose
* Daten von 21 Standorten in Finnland
* Guter Fit durch VARTA
* VARTAX Nutzt zusätzlich noch Monte Carlo Simulation, um sehr schwache und sehr starke Winde mit simulieren zu können.

### Absatz 7

* Vier Ansätze mit ARMA (autoregressive moving average)
* Vorgehen (Nicht so leicht aus der Zusammenfassung zu verstehen, am besten originales Paper lesen):
  1. Geschw.-Vektor wird in Längen/Breitengrad Vektoren zerteilt. 
  2. Beide werden durch ein jeweiliges ARMA Model repräsentiert. Traditionelles ARMA für Geschw. und ein ?verlinktes? Model für Richtung.
  3. VAR Model um die Richtungs/Geschw. Tupel zu berechnen.
  4. ...?
* 1h Horizont
* Bewertet durch MAE
* Ergebnis: Ihr Komponenten-Model ist besser als ein traditionell ?verlinktes? ARMA Model

### Absatz 8

* Vorhandenes Paper: Prognose des stündlichen Mittelwertes mit bis zu 10h Horizont
* Wegen Saisonalen Effekten: 1 Model pr Monat
* Bis zu 20% besser als im Vergleich zu ?vorhandenen? Modellen
* Weitere Erkenntnisse:
  * Wenn Geschw im Bereich von 4-11m/s: Geringer Fehler, wenig abhängig von der Prognoselänge
  * Die größten Fehler treten bei extremen Windgeschwindigkeiten auf (irrelevant, da Turbinen eh ab einer gewissen Stärke abschalten.)