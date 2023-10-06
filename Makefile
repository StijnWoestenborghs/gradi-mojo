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
	. .venv/bin/activate && mojo run main.mojo

cpp-build:
	@echo "Building C++ executable ..."
	g++ -std=c++20 \
		-O3 \
		-o ./cpp/build/bin/gradient_descent \
		-I ./cpp/include \
		./cpp/src/*.cpp

	@echo "Building C++ shared object ..."
	g++ -std=c++20 \
		-O3 \
		-fpic \
		-shared \
		-o ./cpp/build/lib/gradient_descent.so \
		-I ./cpp/include \
		./cpp/src/*.cpp

	@echo "Running C++ executable"
	./cpp/build/bin/gradient_descent