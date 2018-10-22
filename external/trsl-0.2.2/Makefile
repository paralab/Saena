
BUILD_DIR = cmake_build
INSTALL_PREFIX = /usr/local

.PHONY: all build install doc test clean publish

all: build

build: $(BUILD_DIR)/Makefile
	$(MAKE) -C $(BUILD_DIR)

$(BUILD_DIR)/Makefile:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake -DCMAKE_INSTALL_PREFIX=$(INSTALL_PREFIX) ..

install: build
	$(MAKE) -C $(BUILD_DIR) install

doc:
	doxygen doc_src/Doxyfile
	cp LICENSE_1_0.txt documentation/html
	cp doc_src/trsl_logo.jpg documentation/html

test: build
	./$(BUILD_DIR)/test_is_picked_systematic
	./$(BUILD_DIR)/test_random_permutation_iterator
	./$(BUILD_DIR)/test_sort_iterator

clean:
	rm -fr documentation
	rm -fr $(BUILD_DIR)
	rm -fr build

publish: doc
	ssh renauddetry,trsl@shell.sourceforge.net create
	rsync -rl --delete --delete-excluded documentation/html/ renauddetry@shell.sourceforge.net:/home/groups/t/tr/trsl/htdocs
