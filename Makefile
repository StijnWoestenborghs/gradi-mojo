setup: venv
	. .venv/bin/activate && python -m pip install --upgrade pip
	. .venv/bin/activate && pip install -r python-requirements.txt

venv:
	test -d .venv || python3 -m venv .venv

clean:
	rm -rf .venv

py: 
	. .venv/bin/activate && python main.py

mo:
	mojo run main.mojo
