# Step Memoization

## Collocamento

Questa lezione completa il tema dei DAG di workflow introdotto nella
**Lezione 10** (Fan-in/Fan-out e pattern sincroni/asincroni), affrontando la
domanda: *come evitiamo di rieseguire uno step di pipeline il cui input non
e' cambiato?* La seconda parte della lezione allarga lo sguardo a un gruppo
di pattern correlati, che emergono naturalmente non appena la memoization
viene messa alla prova in un contesto distribuito e con step non puramente
funzionali.

## Obiettivi didattici

- Comprendere il pattern architetturale di **step memoization** applicato a
  pipeline ML (non solo a funzioni pure).
- Saper costruire una **cache key (fingerprint)** a partire da input
  eterogenei: tensori PyTorch, iperparametri, codice sorgente dello step.
- Implementare un layer di memoization locale in PyTorch, con backend
  filesystem (`diskcache`) e con backend condiviso (`Redis`, riusando lo
  stack Celery+Redis della Lezione 10).
- Riconoscere i limiti della sola step memoization e i pattern che li
  affrontano:
  - **Cache stampede / dogpile prevention**
  - **Idempotency key pattern**

## Struttura del repository

```
011-step-memoization/
├── README.md
├── memoize_step.py          # decoratore locale, backend diskcache
└── memoize_step_redis.py    # decoratore locale, backend Redis
```

## Come eseguire la demo locale (diskcache)

```bash
pip install torch diskcache
python memoize_step.py
```

Output atteso: la prima esecuzione di `preprocess_dataset` e
`extract_features` produce cache MISS (con tempi di esecuzione simulati di
2s e 3s); la seconda esecuzione con gli stessi argomenti produce cache HIT
istantaneo; una terza esecuzione con `max_len` diverso produce di nuovo un
MISS, perche' la fingerprint dipende anche dagli argomenti di
configurazione, non solo dai dati.

## Come eseguire la demo con Redis

Richiede un'istanza Redis raggiungibile (riusa lo stack della Lezione 10):

```bash
docker run -d -p 6379:6379 redis
pip install torch redis
python memoize_step_redis.py
```

## Pattern correlati

La sola step memoization risolve "non ricalcolare cio' che gia' conosci",
ma apre altri problemi che la lezione affronta a livello concettuale
(senza ancora una implementazione dedicata, che potra' essere oggetto di
un laboratorio successivo):

### Cache stampede / dogpile prevention

Quando piu' worker Celery mancano la cache **nello stesso istante** (tipico
all'avvio di un batch di run identici), tutti ricalcolano lo stesso step in
parallelo, vanificando il beneficio della cache e sovraccaricando la
risorsa condivisa (GPU, servizio esterno, DB). La soluzione tipica e' un
**lock distribuito** (es. `SET key value NX EX ttl` su Redis): il primo
worker che manca la cache acquisisce il lock e calcola, gli altri
attendono o fanno polling; dopo l'acquisizione del lock si ricontrolla la
cache ("double-checked locking"), perche' nel frattempo un altro worker
potrebbe aver gia' terminato.

### Idempotency key pattern

La step memoization evita di ricalcolare un risultato; l'idempotency key
garantisce invece che uno step **con effetti collaterali** (scrittura su
DB, invio di una notifica, submit di un job) produca lo stesso effetto
anche se invocato piu' volte con la stessa chiave, ad esempio a causa di un
retry o di una riesecuzione manuale. E' un pattern complementare, non
alternativo, alla memoization: la fingerprint calcolata per la cache puo'
essere riusata come idempotency key per lo step a valle.