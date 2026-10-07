# Agentic MedMNIST v3

V3 aggiunge un Transformer multi-scala e agenti capaci di creare codice Python
per la singola run. V1 e v2 rimangono indipendenti e non vengono modificate.

## Transformer multi-scala

La famiglia `multi_scale_transformer` combina token di patch a risoluzioni
diverse attraverso un encoder comune. Default: scale `[2, 4, 7]`, larghezza 64,
profondità 2, quattro teste, rapporto MLP 4, dropout 0,1, pooling medio nel
portfolio. Le scale ammesse sono 2, 4, 7 e 14: sceglierne almeno due distinte.
La codifica posizionale è separata per scala, con embedding della scala.
Il modello lavora su immagini RGB 28×28 e produce nove logits.

Il portfolio offline comprende sette famiglie. L'agente può scegliere il modello
multi-scala anche senza abilitare il codice generato; il vincitore dipende dalla
validation. Le configurazioni dei modelli integrati restano utilizzabili con
`lightning_cli.py`.

Da questa directory:

```powershell
python run.py --quick --offline --device cpu --search-trials 7 --search-rounds 2 --search-epochs 1 --max-epochs 2
```

## Feature Pyramid Transformer

La famiglia `feature_pyramid_transformer` implementa ST, GT e RT del
[paper ECCV 2020](https://arxiv.org/abs/2007.09451). ST usa prodotto scalare e
Mixture of Softmaxes con due componenti; GT usa distanza euclidea negativa e
quattro componenti. RT usa pesi di canale da pooling globale e convoluzioni
3×3, con downsampling dalla scala fine alla scala grossolana. Tutti i rami
leggono la piramide originale, prima della concatenazione e fusione 3×3.

Il backbone ResNet-18 compatto, senza pesi preaddestrati, produce livelli
28, 14, 7 e 4 con base width 32. Le proiezioni condividono `hidden` canali
(default 256, multiplo di quattro). Un singolo blocco FPT alimenta pooling
globale dei quattro livelli, concatenazione e classificazione a nove classi.
Il portfolio usa AdamW, batch 32 e dropout 0,1 sulla testa. `depth` viene
canonicalizzato a 1; teste, patch, scale e codifica posizionale sono inattivi.
La fedeltà riguarda gli operatori FPT; backbone, testa e training sono adattati
a PathMNIST e non replicano gli esperimenti COCO del paper.

## Ricerca e training interamente decisi dagli agenti

`--execution-mode agent_autonomous` abilita codice generato sul worker Linux,
richiede Ollama e omette baseline e confronto. Gli agenti scelgono tutti i
parametri sperimentali attivi e decidono quando creare trial, proseguire il
training e terminare la ricerca. Le scelte usano solo training e validation.
Non sono ammessi `--max-epochs`, `--search-epochs`, `--search-trials` o
`--search-rounds` o `--max-generated-bundles`, neppure se coincidono con i default. Non esiste il tetto
di 1000 epoche del percorso precedente. Le decisioni non valide interrompono
la run dopo i tentativi Ollama configurati, senza fallback euristici.

Esempio da eseguire **solo sul controller predisposto per la run**, dopo aver
configurato i segnaposto del worker e di Ollama; questo intervento prepara il
codice e non avvia alcuna run:

```sh
python run.py --execution-mode agent_autonomous --worker-host research-worker --worker-root /srv/agentic-medmnist/jobs --worker-image agentic-medmnist:v3 --device cuda --ollama-base http://localhost:11434 --ollama-model qwen2.5:7b
```

Restano esterni split ufficiali, metriche, protocollo di valutazione, seed e
risorse disponibili: default un'ora per operazione e 16 GiB di memoria.
Non è imposto un limite complessivo di durata della ricerca. I limiti di
trial, round, epoche e versioni del codice del percorso legacy non guidano
questa modalità. Timeout e memoria insufficiente producono segmenti falliti,
con evidenza conservata, e non risultati di training completato.

Ogni segmento salva `model.ckpt` (migliore modello) e `resume.pt` (stato più
recente con optimizer, scheduler, generatori casuali, avanzamento del loader
al confine di epoca, storia ed early stopping). L'azione `continue_trial`
richiede solo identificativo e ulteriori epoche: modifiche a codice o parametri
richiedono un nuovo trial. Uno scheduler OneCycle già esaurito e un trial
fermato dall'early stopping non possono essere estesi senza creare un nuovo
trial. Il periodo iniziale del cosine scheduler viene conservato nella ripresa.

La selezione confronta anche durate diverse mediante accuracy di validation
e macro-F1 nella tolleranza del protocollo. Il checkpoint vincente viene
congelato prima del test, senza retraining aggiuntivo. Le ripetizioni su altri
seed riutilizzano configurazione e sequenza dei segmenti, senza nuove scelte
di ricerca; il replay riproduce le decisioni archiviate senza contattare Ollama.
Report e acceptance dichiarano baseline/confronto non eseguiti: i requisiti
che dipendono dalla baseline non sono attestati. Il percorso `legacy` resta
disponibile con le opzioni precedenti.

Per gli esperimenti autonomi la risposta dell'agente include `training`,
con tutti i controlli attivi espliciti, e un numero positivo di epoche iniziali.
`train(context)` riceve target cumulativo `epochs`, `start_epoch`,
`segment_epochs` e `resume` (percorso input in sola lettura oppure `None`).
La storia restituita è cumulativa; `stop_reason` vale `segment_complete` o
`early_stopping`. Il helper `worker_runtime.fit_model` implementa la ripresa;
i loop personalizzati devono rispettare il contratto di stato descritto nel
prompt, inclusi identificativi della configurazione e dei dati.

## Preparazione del worker Linux

Servono SSH con autenticazione a chiave, host già presente in `known_hosts`,
Python 3 sul worker e Docker Linux con seccomp attivo. Il controller usa i
comandi `ssh` e `scp`; non conserva chiavi o password negli artefatti.

Trasferire Dockerfile e requirements.txt sul worker e costruire l'immagine:

```sh
docker build -t agentic-medmnist:v3 .
mkdir -p /srv/agentic-medmnist/jobs
```

Per CUDA predisporre driver e runtime GPU del worker e un'immagine con PyTorch
CUDA compatibile. `--device cuda` richiede che la GPU sia effettivamente
disponibile nel container. È possibile fornire una propria immagine attraverso
`--worker-image`. Il preflight risolve l'immagine nel suo ID SHA-256 e la run
riutilizza esclusivamente tale identità, senza pull o installazioni implicite.

## Autonomia con codice della run

```powershell
python run.py --quick --allow-generated-code --worker-host research-worker --worker-root /srv/agentic-medmnist/jobs --worker-image agentic-medmnist:v3 --device cpu --search-trials 7 --search-rounds 2 --search-epochs 3 --max-epochs 15 --ollama-base http://localhost:11434 --ollama-model qwen2.5:7b
```

Con `--offline`, questa modalità utilizza un esperimento multi-scala
deterministico e la sua strategia di proposta dei parametri. Ollama aggiunge la
creazione di nuovi sorgenti, la revisione delle ipotesi tra round e fino a due
correzioni automatiche per esperimento. L'orchestrazione resta sequenziale.

Gli agenti possono implementare reti, preprocessing, loss, optimizer, training,
strategie di ricerca e analisi. Il framework conserva gli split ufficiali,
calcola le metriche di selezione sulle predizioni di validation e usa il test
solo dopo il congelamento del vincitore. Baseline, ablation, abstention e OOD
restano disponibili; in modalità autonoma anche training e inferenza della
baseline avvengono sul worker.

Ogni container usa un utente non privilegiato, rete disabilitata, nessuna
capability, `no-new-privileges`, root e input in sola lettura e output dedicati
al job. Repository, cache completa del dataset, credenziali e socket Docker
non sono montati. Ogni job usa un UUID distinto. Se il worker o l'isolamento
non sono disponibili, la modalità autonoma termina: non esiste fallback locale
per il codice generato.

Limiti configurabili: `--worker-timeout` (3600 secondi per trial di ricerca,
condivisi tra verifica, training, inferenza e correzioni; per le altre fasi
il limite si applica alla singola operazione, oltre al tempo di pulizia),
`--worker-memory-gib` (16 GiB), `--max-generated-bundles` (32 versioni per run).
Nel percorso legacy, trial, round ed epoche usano le opzioni esistenti. Il controller controlla gli
orizzonti dichiarati e i risultati; per loop Python personalizzati il timeout
del container è il limite esterno di esecuzione. Le proposte duplicate allo
stesso budget vengono saltate; il numero di trial può essere inferiore al
massimo. Il round finale confronta solo trial riusciti con lo stesso budget
allocato. Se tutti falliscono, la selezione termina con errore.

## Interfaccia degli esperimenti

Il bundle contiene `experiment.py` ed eventuali moduli Python ausiliari:

```python
def build_model(context): ...  # torch.nn.Module; input float [N,3,28,28], output [N,9]
def train(context): ...        # salva /output/model.ckpt e restituisce storia e riepilogo
def predict(context): ...      # carica context['checkpoint']; restituisce numpy float [N,9]
def propose(context): ...      # opzionale: suggerimenti di parametri per la ricerca
```

`context` contiene `config`, `parameters`, `data`, `output`, `checkpoint`,
`epochs` ed `evidence`. Train riceve gli array `train_images`, `train_targets`,
`val_images`, `val_targets`; l'inferenza riceve solo `images`, mai etichette.
Le immagini sono NCHW float, con la rappresentazione selezionata applicata dal
framework; il codice può aggiungere un proprio preprocessing. I parametri
appresi devono essere salvati nel checkpoint e riutilizzati in inferenza.

`train` restituisce `final_train_loss`, `final_val_accuracy`,
`best_val_accuracy`, `best_epoch`, `epochs_completed`, `history`. Ogni riga di
`history` contiene `epoch`, `train_loss`, `val_accuracy` e può includere
`val_loss`, `val_macro_f1`, `learning_rate`. Epoche contigue da 1, valori finiti
e orizzonte entro `context['epochs']` sono obbligatori. Questi valori sono
diagnostici: il ranking usa metriche ricalcolate dal controller.

`propose` riceve solo risultati precedenti di validation, senza dataset, e
restituisce una lista di oggetti con `parameters`, ed eventualmente `epochs`
e `batch_size`. Il controller valida le proposte e conserva i budget. Si può
usare `worker_runtime.fit_model` oppure scrivere interamente il training.
`default_experiment.py` documenta un esempio eseguibile nel container.

## Artefatti, replay e verifiche

I sorgenti vengono salvati in `seed_<seed>/generated/<bundle_id>/vNNN/`, con
manifest e checksum per ogni file. Le correzioni creano versioni nuove. Il
controller analizza la sintassi ma non importa mai il codice generato né
deserializza i suoi checkpoint. I job, gli input, i log e gli output sono in
`blobs/worker_jobs/`; anche il worker conserva i job per audit. La pulizia di
questi archivi è esplicita e non viene eseguita automaticamente.

Il vincitore esportato include il bundle. `best_config.yaml` degli esperimenti
generati usa il formato JSON-compatible `isolated-experiment-v1`, non il formato
LightningCLI. Per rieseguire il vincitore usare:

```powershell
python generated_cli.py fit --config runs/pathmnist_ID/best_task_config.yaml --run-root runs/pathmnist_ID --output-root runs/retrained --data-root PATH --device cpu
```

I seed successivi copiano e verificano il bundle congelato, senza nuove
proposte. Per riprodurre l'intera run con le decisioni archiviate:

```powershell
python run.py --replay-run runs/pathmnist_ID/seed_42 --output-root runs/replayed
python -m unittest discover -s tests -v
```

I test locali verificano il modello con training reale, checkpoint e replay,
l'integrità dei bundle e il flusso autonomo attraverso un worker simulato.
Per i test reali di isolamento impostare `AGENTIC_TEST_WORKER_HOST`,
`AGENTIC_TEST_WORKER_ROOT`, `AGENTIC_TEST_WORKER_IMAGE` ed eventualmente
`AGENTIC_TEST_WORKER_DEVICE` (default CPU), poi eseguire
`python -m unittest tests.test_worker_integration -v`. Senza queste variabili
i test remoti sono esplicitamente saltati.
