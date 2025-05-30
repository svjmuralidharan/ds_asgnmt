# Name of your docker image
IMAGE = community_analysis

# Default target
.PHONY: all
all: run

# Build Docker image
.PHONY: build
build:
	docker build -t $(IMAGE) .

# Run analysis (main script)
.PHONY: run
run: build
	docker run --rm \
		-v "$(PWD)/data":/app/data \
		-v "$(PWD)/output":/app/output \
		-v "$(PWD)/figures":/app/figures \
		-v "$(PWD)/src":/app/src \
		-v "$(PWD)":/app \
		$(IMAGE)

# Run pytest tests in Docker
.PHONY: test
test: build
	docker run --rm \
		-v "$(PWD)/src":/app/src \
		-v "$(PWD)/test":/app/test \
		-v "$(PWD)/data":/app/data \
		-v "$(PWD)/output":/app/output \
		-v "$(PWD)/figures":/app/figures \
		-v "$(PWD)":/app \
		$(IMAGE) pytest test/

# Remove output and figure files
.PHONY: clean
clean:
	rm -rf output/*.csv output/*.md figures/*.png

