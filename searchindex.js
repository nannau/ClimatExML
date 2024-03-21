Search.setIndex({"docnames": ["build_docs", "configuring", "customizing", "infering", "installing", "intro", "performance", "tracking", "training"], "filenames": ["build_docs.md", "configuring.md", "customizing.md", "infering.md", "installing.md", "intro.md", "performance.md", "tracking.md", "training.md"], "titles": ["How to build docs", "Configuring MLflow", "Customize", "Preprocessing Pipeline", "Installing ClimatExML", "ClimatExML Documentation", "Performance", "Comet", "Training"], "terms": {"http": [0, 1, 4, 5], "jupyterbook": 0, "org": [0, 5], "en": 0, "stabl": 0, "publish": [], "gh": 0, "page": 0, "html": 0, "creat": [0, 1, 7], "document": [0, 1, 7], "virtual": 0, "we": [0, 1], "call": [0, 1, 7], "jbook": 0, "m": [0, 1, 4], "venv": [0, 1, 4], "i": [0, 1, 4, 5, 7, 8], "project": [1, 5, 7, 8], "allow": [1, 7, 8], "automat": 1, "log": 1, "displai": 1, "data": [1, 5, 8], "about": 1, "your": [1, 4, 7], "ml": [1, 8], "model": [1, 5], "train": [1, 4, 5, 7], "algorithm": [1, 5], "live": 1, "It": [1, 4, 5, 7, 8], "critic": 1, "compon": [1, 5], "evalu": 1, "differ": 1, "debug": [1, 8], "issu": 1, "code": [1, 4, 8], "climatexml": [0, 1, 7, 8], "us": [1, 4, 5, 7, 8], "metric": 1, "gener": [1, 5], "artifact": 1, "like": [1, 4], "plot": 1, "imag": 1, "well": [1, 4], "save": 1, "should": [1, 4, 7], "refer": [1, 7], "so": [1, 4, 8], "thei": 1, "can": [1, 4, 7, 8], "fine": 1, "tune": 1, "match": 1, "need": [1, 4, 8], "thi": [1, 4, 5, 7, 8], "provid": [1, 4, 7], "quick": 1, "overview": 1, "two": [1, 4], "main": [1, 4], "an": [1, 4, 7], "s3": 1, "bucket": 1, "hpc": [1, 4], "system": [1, 4, 5, 8], "without": 1, "internet": 1, "access": 1, "while": [1, 4], "requir": [1, 4], "postgr": 1, "databas": 1, "extern": 1, "also": [1, 4, 8], "object": 1, "store": 1, "setup": 1, "If": [1, 4, 7], "you": [1, 4, 7], "ar": [1, 4, 7, 8], "member": 1, "uvic": 1, "climat": [1, 5], "lab": 1, "eccc": 1, "send": 1, "email": 1, "request": 1, "convenei": 1, "persist": [1, 4], "storag": 1, "arbutu": 1, "which": [1, 4, 7, 8], "here": [1, 4, 7], "cloud": 1, "computecanada": 1, "ca": 1, "make": [1, 4, 8], "There": [1, 4], "openstack": 1, "manag": 1, "resourc": 1, "run": [1, 4, 8], "must": [1, 7], "have": [1, 7, 8], "instal": [0, 1, 5, 7, 8], "boto3": 1, "manual": 1, "add": 1, "other": [1, 4, 7], "ssh": 1, "t": [1, 8], "do": [1, 8], "from": [0, 1, 4], "environ": [1, 7, 8], "variabl": [1, 5, 7], "file": [1, 7, 8], "set": 1, "env": 1, "export": [1, 7], "mlflow_tracking_uri": 1, "public": 1, "ip": 1, "5000": 1, "mlflow_s3_endpoint_url": 1, "aws_access_key_id": 1, "get": [1, 4, 7], "aws_secret_access_kei": 1, "To": [0, 1], "first": [1, 7], "download": 1, "opensrc": 1, "sh": [1, 8], "client": 1, "correspond": 1, "specif": 1, "allcoat": 1, "name": [1, 4, 7], "sourc": [0, 1, 4, 7], "openrc": 1, "ec2": 1, "credenti": 1, "want": [1, 7], "secret": 1, "info": 1, "bash": [1, 7, 8], "same": 1, "Then": [1, 4], "uri": 1, "postgresql": 1, "usernam": 1, "password": 1, "localhost": 1, "5432": 1, "mlflowdb": 1, "host": [1, 4, 8], "0": [1, 4], "default": [1, 4], "root": [1, 4], "name_of_bucket": 1, "beforehand": 1, "note": [1, 7, 8], "These": 1, "instruct": [1, 4, 7], "process": [1, 4, 5], "similar": 1, "start": [0, 1, 4, 7], "fresh": 1, "python": [1, 4], "virtualenv": 1, "bin": [0, 1, 4], "activ": [0, 1, 4], "pip": [0, 1, 4, 7], "index": 1, "upgrad": 1, "modul": 1, "load": [1, 4], "gcc": 1, "9": [1, 4], "3": 1, "arrow": 1, "8": 1, "now": 1, "command": [1, 7], "built": [0, 4], "user": [4, 7, 8], "modifi": 4, "suit": [4, 8], "prefer": 4, "wai": [1, 4, 7], "By": 4, "local": 4, "machin": [4, 7], "contain": [1, 4], "design": [4, 5], "option": 4, "1": [4, 5], "slightli": 4, "more": [1, 4, 7], "configur": [4, 5, 7, 8], "simpler": 4, "quickli": 4, "2": [4, 5], "highli": 4, "portabl": 4, "conveni": 4, "pipelin": [4, 5], "both": [1, 4], "recommend": 4, "basic": [4, 7], "includ": [1, 4], "below": [4, 7], "climatexvenv": 4, "begin": 4, "clone": 4, "repo": [0, 4], "git": [0, 4], "github": 4, "com": 4, "nannau": [4, 8], "packag": [4, 7], "e": 4, "necessari": 4, "sure": [1, 4], "work": [4, 8], "correctli": [4, 7], "smi": 4, "return": [4, 7], "someth": 4, "tue": 4, "jan": 4, "14": 4, "30": 4, "42": 4, "2024": 4, "535": 4, "129": 4, "03": 4, "driver": 4, "version": 4, "cuda": 4, "12": 4, "bu": 4, "id": 4, "disp": 4, "A": 4, "volatil": 4, "uncorr": 4, "ecc": 4, "fan": 4, "temp": 4, "perf": 4, "pwr": 4, "usag": 4, "cap": 4, "memori": 4, "util": 4, "comput": 4, "mig": 4, "geforc": 4, "rtx": 4, "xxxx": 4, "off": 4, "00000000": 4, "01": 4, "00": 4, "On": 4, "35c": 4, "p8": 4, "21w": 4, "450w": 4, "61mib": 4, "24564mib": 4, "n": [0, 4], "gi": 4, "ci": 4, "pid": 4, "type": 4, "2020": 4, "g": 4, "usr": 4, "lib": 4, "xorg": 4, "35mib": 4, "2079": 4, "gnome": 4, "shell": [4, 7, 8], "13mib": 4, "": 4, "import": [0, 1, 4, 7], "function": [4, 7], "expect": 4, "That": 4, "abl": [4, 7], "commun": 4, "tensor": 4, "etc": 4, "fortun": 4, "easi": [1, 4], "check": 4, "torch": 4, "is_availab": 4, "true": 4, "As": [4, 7], "extra": 4, "step": 4, "onto": 4, "randn": 4, "1000": 4, "error": 4, "rais": 4, "oper": 4, "encount": 4, "troubl": 4, "base": 4, "outsid": 4, "describ": 5, "softwar": 5, "statist": 5, "downscal": 5, "implement": 5, "super": 5, "resolut": 5, "adversari": 5, "network": 5, "srgan": 5, "follow": 5, "acm23": 5, "The": [1, 5, 7], "optim": 5, "convect": 5, "permit": 5, "high": 5, "target": 5, "condit": 5, "low": 5, "input": 5, "convolut": 5, "neural": 5, "cnn": 5, "predict": 5, "multivari": 5, "fire": 5, "weather": 5, "precipit": 5, "humid": 5, "surfac": 5, "temperatur": 5, "wind": 5, "support": 5, "climatex": [5, 7], "mlflow": [5, 7, 8], "custom": 5, "preprocess": 5, "perform": 5, "nicolaa": 5, "j": 5, "annau": 5, "alex": 5, "cannon": 5, "adam": 5, "h": 5, "monahan": 5, "hallucin": 5, "scale": 5, "artifici": 5, "intellig": 5, "earth": 5, "4": 5, "e230015": 5, "2023": 5, "url": 5, "journal": 5, "ametsoc": 5, "view": 5, "ai": 5, "d": 5, "23": 5, "0015": 5, "xml": 5, "doi": 5, "10": 5, "1175": 5, "polici": 0, "checkout": 0, "docs_sourc": 0, "befor": 0, "develop": 0, "In": [0, 7], "subdirectori": 0, "ha": [0, 8], "relev": [0, 7], "inform": 0, "edit": [0, 7, 8], "valu": [0, 7], "branch": 0, "ghp": 0, "push": 0, "p": 0, "f": 0, "_build": 0, "how": 5, "build": 5, "doc": [5, 7], "non": 1, "next": [1, 7], "similarli": 1, "time": [1, 7], "chang": [1, 8], "sens": 1, "mlflow_sqlite_db_path": 1, "path": 1, "db": 1, "mlflow_artifacts_path": 1, "ident": 1, "replic": [1, 8], "choos": [1, 8], "logic": 1, "locat": 1, "extens": 1, "number": 1, "slash": 1, "exist": 1, "otherwis": 1, "launch": 1, "doesn": [1, 8], "perman": 1, "just": [1, 8], "serv": [1, 7], "onc": 1, "successfulli": 1, "ctrl": 1, "c": 1, "close": 1, "pre": 8, "depend": 8, "normal": 8, "directori": 8, "repres": 8, "insid": 8, "easier": 8, "within": 8, "framework": 8, "rather": 8, "than": 8, "copi": 8, "all": 8, "them": 8, "track": [7, 8], "simpli": 8, "execut": 8, "srun": 8, "exec": 8, "home": 8, "nv": 8, "scratch": 8, "marvin_light_contain": 8, "slurm_tmpdir": 8, "light_contain": 8, "overlai": 8, "lightn": 8, "sif": 8, "apptainer_cmd": 8, "my": 8, "own": 8, "fakeroot": 8, "sr": 8, "script": 8, "yet": 8, "v": 8, "container_cmd": 8, "rm": 8, "runtim": 8, "nvidia": 8, "gpu": 8, "project_dir": 8, "data_dir": 8, "tool": 7, "learn": 7, "experi": 7, "real": 7, "fairli": 7, "straightforward": 7, "onli": 7, "detail": 7, "previous": 7, "found": 7, "md": 7, "sign": 7, "write": 7, "complet": 7, "free": 7, "though": 7, "paid": 7, "plan": 7, "advanc": 7, "desir": 7, "after": 7, "api": 7, "applic": 7, "program": 7, "interfac": 7, "kei": 7, "form": 7, "authent": 7, "go": 7, "account": 7, "select": 7, "keep": 7, "secur": 7, "comet_ml": 7, "comet_api_kei": 7, "whose": 7, "done": 7, "via": 7, "your_api_key_her": 7, "doe": 7, "expir": 7, "end": 7, "session": 7, "redefin": 7, "futur": 7, "instead": 7, "append": 7, "rc": 7, "bashrc": 7, "echo": 7, "either": 7, "section": 7, "conf": 7, "config": 7, "yaml": 7}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"how": 0, "build": 0, "doc": 0, "jupyt": 0, "book": 0, "python": 0, "environ": [0, 4], "configur": 1, "mlflow": 1, "option": 1, "1": 1, "remot": 1, "instanc": 1, "track": 1, "server": 1, "kei": 1, "2": 1, "On": [1, 8], "allianc": [1, 8], "machin": [1, 8], "spin": 1, "up": [1, 7, 8], "user": 1, "interfac": 1, "sqlite": 1, "backend": 1, "custom": 2, "preprocess": 3, "pipelin": 3, "instal": 4, "climatexml": [4, 5], "virtual": 4, "verifi": 4, "nvidia": 4, "gpu": 4, "hardwar": 4, "pytorch": 4, "access": 4, "document": 5, "perform": 6, "run": 7, "digit": 8, "research": 8, "publish": 0, "train": 8, "contain": 8, "set": [7, 8], "bind": 8, "apptain": 8, "docker": 8, "without": 8, "comet": 7}, "envversion": {"sphinx.domains.c": 3, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 9, "sphinx.domains.index": 1, "sphinx.domains.javascript": 3, "sphinx.domains.math": 2, "sphinx.domains.python": 4, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinxcontrib.bibtex": 9, "sphinx": 60}, "alltitles": {"How to build docs": [[0, "how-to-build-docs"]], "Jupyter Book": [[0, "jupyter-book"]], "Python Environment": [[0, "python-environment"]], "Building the docs": [[0, "building-the-docs"]], "Publishing docs": [[0, "publishing-docs"]], "Configuring MLflow": [[1, "configuring-mlflow"]], "Option 1: Remote MLflow Instance": [[1, "option-1-remote-mlflow-instance"]], "Remote Tracking Server": [[1, "remote-tracking-server"]], "Keys": [[1, "keys"]], "Option 2: On Alliance Machines": [[1, "option-2-on-alliance-machines"]], "Spin up MLflow User Interface with SQlite backend": [[1, "spin-up-mlflow-user-interface-with-sqlite-backend"]], "Customize": [[2, "customize"]], "Preprocessing Pipeline": [[3, "preprocessing-pipeline"]], "Installing ClimatExML": [[4, "installing-climatexml"]], "Installation in a Virtual Environment": [[4, "installation-in-a-virtual-environment"]], "Verify NVidia GPU Hardware": [[4, "verify-nvidia-gpu-hardware"]], "Verify PyTorch + GPU Access": [[4, "verify-pytorch-gpu-access"]], "ClimatExML Documentation": [[5, "climatexml-documentation"]], "Performance": [[6, "performance"]], "Comet": [[7, "comet"]], "Setting Up": [[7, "setting-up"]], "Running": [[7, "running"]], "Training": [[8, "training"]], "Containers": [[8, "containers"]], "Setting up Binds": [[8, "setting-up-binds"]], "Apptainer": [[8, "apptainer"]], "Digital Research Alliance Machines": [[8, "digital-research-alliance-machines"]], "Docker": [[8, "docker"]], "Without Containers": [[8, "without-containers"]], "On Digital Research Alliance Machines": [[8, "on-digital-research-alliance-machines"]]}, "indexentries": {}})