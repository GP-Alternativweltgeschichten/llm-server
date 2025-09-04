# LLM-Server für das Alternate-History-Projekt

## Beschreibung

Dieser Server stellt eine REST-API bereit, die es ermöglicht, verschiedene Large Language Models (LLMs) für spezifische
Aufgaben im Rahmen eines Alternate-History-Projekts zu nutzen. Die Hauptfunktion des Servers ist die Unterstützung der
Stadtplanung und 3D-Modellgenerierung. Über die API können Aktionen angefordert werden, um eine virtuelle Stadt zu
modifizieren, beispielsweise durch das Hinzufügen oder Entfernen von Gebäuden. Zusätzlich ist der Server in der Lage,
Blender-kompatible Python-Skripte sowie 3D-Modelle zu generieren. Das System unterstützt sowohl lokale
Modelle wie Mistral und LLaMA als auch die Anbindung an die OpenAI-API für die Nutzung von ChatGPT. Für die Generierung
von 3D-Modellen wird das Modell Shape-E von OpenAI genutzt.

## Projektstruktur

```
app.py                 # Hauptserver mit Flask-API (Status, Textgenerierung, Blender-Code-Generierung)
model_registry.py      # Modell-Factory zur Auswahl und Initialisierung der Sprachmodelle
/models                # Implementierungen der unterstützten Modelle
  chatgpt.py           # ChatGPT-Integration (OpenAI API)
  llama_32.py          # LLaMA 3.2 (lokal, Huggingface/Transformers)
  mistral.py           # Mistral 7B (lokal, Mistral Libraries)
  shap_e.py            # 3D-Modellgenerierung (Shape-E Anbindung)
output/                # Ablage generierter 3D-Modelle
shap_e_model_cache/    # Modell- und Konfigurationsdateien für Shape-E
static/                # (optional) Statische Dateien für ein Web-Frontend
requirements.txt       # Python-Abhängigkeiten
README.md              # Projektdokumentation
.env.example           # Beispiel für Umgebungsvariablen
```

## Funktionale Inhalte

### LLM-Integration

Der Server ermöglicht die Auswahl eines spezifischen LLMs zur Verarbeitung der Anfragen. Dies geschieht über die
Umgebungsvariable `MODEL_NAME`, die entweder auf "mistral", "llama_32" oder "chatgpt" gesetzt werden kann. Wird chatgpt
gewählt,
ist ein gültiger API-Schlüssel in der Umgebungsvariable `OPENAI_API_KEY` zwingend erforderlich. Die `model_registry.py`
fungiert als zentrale Fabrik, die das ausgewählte Modell initialisiert und eine einheitliche Schnittstelle bereitstellt.
Angesteuert werden die Modelle immer mit einem System Prompt, um Kontext und Verhalten zu steuern und die Nutzereingaben
zu guardrailen. Dieser findet sich in der Datei `system_prompt.txt` und kann dort angepasst werden werden.

### API-Endpunkte

* `GET /status`
  Liefert Basisinformationen zum Serverstatus (z. B. Verfügbarkeit, ausgewähltes Modell).
* `POST /generate`
  Erwartet JSON-Eingaben zur Erzeugung von Aktionen oder Textausgaben für die Stadtplanung; liefert JSON-Aktionen oder
  Texte zurück.
* `POST /generate_blender_code`
  Erzeugt Blender-kompatible Python-Skripte; optional: führt das Skript auf dem Server aus und legt erzeugte 3D-Objekte
  in `output/` ab.
    * **Hinweis**: Dieser Endpunkt wird aktuell nicht genutzt aufgrund mangelnder Qualität der Ergebnisse.
* `POST /generate_3d_model`
  Nimmt Textbeschreibungen entgegen, generiert 3D-Modelle via Shape-E, optimiert das 3D-Modell durch Reduktion der
  Vertices für bessere Performance in z.B. Unity und sendet es als Wavefront .obj-Datei zurück.

### 3D-Modellgenerierung

Die 3D-Modellgenerierung verwendet das Shape-E-Modell von OpenAI. Die
generierten Modelle sind mit Blender kompatibel und werden automatisch im Verzeichnis output/ als .obj-Datei
gespeichert.    
Da die generierten Modelle sehr groß sein könnten, wird mithilfe der Bibliothek `pymeshlab` und dem
Filter `meshing_decimation_quadric_edge_collapse` die Anzahl an Vertices reduziert und so die Datei verkleinert, jedoch
mit Qualitätsverlust.
Diese Datei wird an den Nutzer zurückgegeben. Im Kontext des Alternate-History-Projekts werden diese Modelle zur
Laufzeit in Unity integriert.

### Konfiguration

Die gesamte Konfiguration des Servers erfolgt über eine .env-Datei, die als Template aus der .env.example-Datei erstellt
werden kann. Zu den wichtigsten Konfigurationsvariablen gehören MODEL_NAME zur Auswahl des gewünschten LLMs sowie
OPENAI_API_KEY für die Nutzung von ChatGPT. Abhängig von den verwendeten Modellen können weitere Variablen wie spezielle
Tokens (z.B. Huggingface) erforderlich sein.

## Nicht-funktionale Anforderungen

* Zielplattform: Python 3.8+
* API-Framework: Flask (in `app.py`)
* Trennung von Modell-Logik (`/models`) und Server-Logik (`app.py`, `model_registry.py`)
* Erwartete Hardware: Lokale Modelle benötigen ausreichend VRAM; Cloud-Anbindung entfällt nur bei `chatgpt`.

## Setup

1. Voraussetzungen:

* Python 3.8 oder höher
* pip

2. Abhängigkeiten installieren:

   ```bash
   pip install -r requirements.txt
   ```

3. `.env` anlegen:

   ```bash
   cp .env.example .env
   ```

   In `.env` die benötigten Werte eintragen (z. B. `MODEL_NAME`, `OPENAI_API_KEY`).

4. (Optional) Modelle und Tokens bereitstellen:

* Huggingface-Token für Zugriff auf Modelle (falls erforderlich)

5. Server starten:

   ```bash
   python app.py
   ```

## Ablauflogik (Kurzbeschreibung)

1. Beim Start liest `model_registry.py` die Umgebungsvariable `MODEL_NAME` und initialisiert das entsprechende Modell.
2. API-Anfragen werden in `app.py` verarbeitet; je nach Endpunkt werden Text, Aktionen oder Blender-Code erzeugt.
3. Für 3D-Anfragen wird Shape-E genutzt; Ergebnisse werden in `output/` gespeichert und können
   per `generate_blender_code` weiterverarbeitet werden.
4. Persistenz für generierte Artefakte erfolgt durch das Dateisystem; Logging erfolgt in der Serverausgabe (
   erweiterbares Logging-Framework empfohlen).

## Sicherheit und Betrieb

* Sensible Informationen (API-Keys, Tokens) dürfen nur in `.env` gespeichert werden und müssen in Versionskontrolle
  ausgeschlossen werden (`.gitignore`).
* Zugriffssteuerung auf den Server (z. B. Reverse Proxy, Authentifizierung) ist projektspezifisch zu implementieren.
* Bei Betrieb lokaler Modelle ist Ressourcennutzung (GPU/CPU, RAM) zu überwachen.

## Fehlerbehandlung & Logging (aktuell)

Das System verfügt über eine grundlegende Fehlerbehandlung in den Flask-Handlern, die entsprechende HTTP-Statuscodes und
Fehlermeldungen zurückgibt. Für eine robustere Produktionsempfehlung könnte die Fehlerbehandlung um ein
strukturiertes Logging-Framework erweitert werden, das auch Rotation von Logdateien und zentrale Fehlerreports
unterstützt.

## Ausblick

Die zukünftige Entwicklung könnte die Unterstützung weiterer oder neuerer LLMs (z.B. DeepSeek) und
Text-to-3D-Generatoren (z.B. [stable-zero123](https://huggingface.co/stabilityai/stable-zero123)) beinhalten, um die
Flexibilität des Servers zu
erhöhen und die Qualität des Outputs zu verbessern. Des Weiteren ist eine robustere Fehlerbehandlung und ein erweitertes
Logging sinnvoll. Optional könnte ein
Web-Frontend zur interaktiven Visualisierung und Steuerung hinzugefügt werden. Es gibt auch Pläne, die 3D-Pipeline zu
erweitern, zum Beispiel durch die Automatisierung von Nachbearbeitungsschritten in Blender.

## Hinweise zur Weiterentwicklung

Für eine nachhaltige Weiterentwicklung ist es ratsam, die Modell-Factory so zu gestalten, dass neue
LLM-Implementierungen mit minimalem Aufwand integriert werden können. Die Reproduzierbarkeit von Ergebnissen sollte
durch die Versionierung von Modellgewichten und Konfigurationen sichergestellt werden. Um die Performance bei langen
Generierungsaufgaben zu verbessern, sollte die Implementierung von Batch-Verarbeitung und Queuing-Systemen in Betracht
gezogen werden.

