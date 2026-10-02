# Protocollo di consolidamento e validazione PathMNIST

Data: 2026-10-02. Run esplorativa di riferimento: `pathmnist_20261001T175019922973Z`.

## Stato ed evidenze da conservare

La run storica rimane immutata. L'audit successivo è salvato in
`v2/audits/pathmnist_20261001T175019922973Z/`. Il codice corrente e l'ambiente
Windows/CPU differiscono dall'ambiente Linux/GPU storico: test del codice
corrente, inferenza dai checkpoint e replay del training sono evidenze distinte.
Non aggiornare retroattivamente l'accettazione storica con test di un altro codice.

Prima di nuovi training archiviare codice esatto, dipendenze, versione CUDA/cuDNN,
hardware, dataset e relativo checksum, digest del modello Ollama, cache delle
fonti e protocollo. Conservare separatamente le evidenze binarie: `*.npz` e
`*.ckpt` sono ignorati da Git. Registrare hash e posizione dell'archivio esterno.

Il test di questa run è già stato esaminato. Le nuove analisi sullo stesso test
sono approfondimenti del benchmark, non conferma su un holdout mai osservato.
Non scegliere configurazioni, soglie o metodi in base al miglior seed di test.
Per una conferma indipendente utilizzare un'ulteriore coorte compatibile, se disponibile.

## A. Ripetibilità a configurazione congelata

Obiettivo: separare la variabilità del training da quella della ricerca.
Nel prossimo esperimento eseguire la selezione sulla validation una sola volta,
poi congelare architettura, rappresentazione, iperparametri e budget finale per
i seed 42, 47 e 72; cambiare solo il seed. Tre seed sono il minimo descrittivo,
non una prova statistica forte. Non usare `--search-per-seed`.

Il runner corrente congela la configurazione del primo seed per i successivi.
Il `best_task_config.yaml` storico è invece un risultato esplorativo selezionato
tra ricerche separate. Non presentare una nuova ricerca come replica esatta di
quel file. Se si vuole ripetere proprio tale configurazione, aggiungere prima
un ingresso esplicito e validato per `BestConfiguration`, inclusa la rappresentazione;
non abusare di `--replay-run`, che richiede il codice originale.

Misurare media e deviazione standard campionaria di accuracy, macro-F1 e
balanced accuracy; pubblicare risultati individuali e recall per classe.

## B. Ablation della rappresentazione

Congelare modello, seed, optimizer, scheduler e budget. Confrontare:

1. unit scaling, senza augmentation;
2. standardizzazione train-only, senza augmentation;
3. standardizzazione train-only più hflip;
4. rappresentazione completa selezionata sulla validation.

Le prime tre condizioni sono già implementate da `--ablation-suite`; la quarta
è la pipeline principale. Registrare esempi effettivi processati: lo stesso
numero di epoche con augmentation materializzate non è lo stesso costo.
Non interpretare queste condizioni come ablation dell'LLM.

## C. Contributo della ricerca e del reasoning

Prima dell'esecuzione implementare e verificare le condizioni seguenti. Il runner
attuale non offre ancora tutti questi controlli; non inventare flag mancanti.

| Condizione | Controllo richiesto |
|---|---|
| Ricerca agentica completa | LLM e cache di letteratura congelati |
| Ricerca convenzionale | Stesso spazio, famiglie, schedule di budget e selettore, proposte casuali con seed registrato |
| Fallback euristico | Stesso spazio e budget della ricerca agentica, nessuna chiamata LLM |
| Senza Prior-Art Scout | LLM uguale, contesto bibliografico rimosso; dati e budget invariati |
| Controllo fisso ResNet18 | Architettura comparabile, rappresentazione e training dichiarati |

Il flag `--offline` corrente cambia insieme LLM e retrieval: da solo non isola
l'effetto del reasoning. Congelare lo stesso corpus per il confronto LLM/euristiche.
La Tiny CNN storica resta un controllo debole, non il solo confronto scientifico.

Registrare trial tentati/completati/falliti, epoche ed esempi processati, GPU-secondi,
tempo totale, numero di parametri, chiamate e latenza LLM. Dichiarare prima
il budget di proposta e il budget di training; confrontare anche i costi misurati.
Preferire differenze appaiate sui seed. Eventuali bootstrap sui campioni del
benchmark devono dichiarare il grain: senza identità di paziente/slide non sono
intervalli di generalizzazione clinica indipendente.

## D. Astensione e OOD

Selezionare score, calibration e soglie solo sulla validation. Applicarle immutate
al test; pubblicare coverage, rischio selettivo, numero di casi da revisionare e
coverage/recall per classe. L'80% è un target di validation, non una coverage
garantita sul test. La coda persistita dimostra handoff, non revisione umana conclusa.

Per OOD pubblicare ogni severità, ogni score e il minimo AUROC, oltre alla media.
La media è aritmetica degli AUROC delle severità per lo score selezionato; non è
un AUROC pooled. Il pass attuale (ogni severità almeno 0,5) è un controllo minimo,
non un requisito sufficiente di robustezza.

Investigazione del seed 72: verificare pipeline del rumore/normalizzazione,
distribuzione dei logits e classi predette, confidence sulle corruzioni e score
alternativi sulla validation. Non invertire lo score scegliendo il verso dal test.
Non correggere il report nascondendo la severità 0,4. Aggiungere altri shift solo
con protocollo dichiarato; il rumore gaussiano resta un proxy, non OOD clinico.

## E. Governance e presentazione

Un warning richiede una risoluzione documentata oppure una mitigazione nominativa
approvata. Un commento LLM non è di per sé prova di violazione. Separare primitive
integrate da nuove estensioni, parametri attivi da campi comuni inattivi e
tracciabilità delle citazioni da supporto scientifico delle inferenze.

Sul codice corrente misurare fault injection e test. Per il replay storico
recuperare la versione originale che corrisponde al checksum del manifest e il
suo ambiente. Una prova sintetica del replay non chiude il replay di questa run.
La dimostrazione e il sign-off indipendenti restano attività umane da documentare.

La presentazione può dichiarare MVP completato e guadagno rispetto al controllo
fisso; deve qualificare i KPI di fault detection con la versione verificata.
Riproducibilità completa, contributo isolato dell'orchestrazione e TRL 7 rimangono
obiettivi aperti fino alla disponibilità delle rispettive evidenze.

## Comandi immediatamente supportati

Eseguire dalla directory `v2`, con l'interprete del progetto:

```powershell
& '../.venv/Scripts/python.exe' audit_run.py runs/pathmnist_20261001T175019922973Z --output audits/pathmnist_20261001T175019922973Z/audit.json
& '../.venv/Scripts/python.exe' check_checkpoint_evidence.py runs/pathmnist_20261001T175019922973Z --output audits/pathmnist_20261001T175019922973Z
& '../.venv/Scripts/python.exe' governance.py validate --output audits/pathmnist_20261001T175019922973Z/current_source_validation.json
```

Sulla macchina GPU, dopo aver validato e archiviato il codice finale, il comando
seguente avvia **un nuovo esperimento**, con configurazione congelata dopo il
primo seed e ablation della rappresentazione. Rigenerare `validation_evidence.json`
su quel codice; usare il nome effettivo del modello Ollama e registrarne il digest.

```bash
python governance.py validate --output validation_evidence.json
python run.py --seeds 42,47,72 --device cuda --max-epochs 15 --search-trials 8 --search-rounds 3 --search-epochs 3 --ablation-suite --require-llm --ollama-base http://localhost:11434 --ollama-model qwen3.8 --literature-cache runs/pathmnist_20261001T175019922973Z/literature_cache --validation-evidence validation_evidence.json --output-root runs_validation
```

Il corpus esistente viene riutilizzato; non è una nuova ricerca web. Questo
esperimento copre A e B, non sostituisce i confronti ancora da implementare in C.
