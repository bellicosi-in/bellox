#include "common.h"
#include "chunk.h"
#include "value.h"
#include "debug.h"
#include "memory.h"
#include "vm.h"
#include "compiler.h"

static char* readFile(const char* path){
  FILE* file = fopen(path,"rb");
  if(file == NULL){
    fprintf(stderr, "could not open the file. [%s]", path);
    exit(60);
  }
  fseek(file, 0L, SEEK_END);
  size_t fileSize = ftell(file);
  rewind(file);

  char* buffer = (char*)malloc(fileSize + 1);
  if(buffer==NULL){
    fprintf(stderr, "could not allocate space [%s]", path);
    exit(60);
  }
  size_t bytesRead = fread(buffer, sizeof(char), fileSize, file);
  if(bytesRead < fileSize){
    fprintf(stderr, "could not read the entire file [%s]", path);
    exit(60);
  }
  buffer[fileSize] = '\0';
  fclose(file);
  return buffer;
}

static void runFile(const char* path){
  char* source = readFile(path);
  InterpretResult result = interpret(source);
  free(source);
  if(result == INTERPRET_COMPILE_ERROR) exit(74);
  if(result == INTERPRET_RUNTIME_ERROR) exit(75);

}

static void repl(){
  char line[1024];
  for(;;){
    printf("> ");
    if(!fgets(line, sizeof(line), stdin)){
      printf("\n");
      break;
    }
    interpret(line);
  }
}


int main(int argc, char* argv[]){
  initVM();
  
  if(argc == 1){
    repl();
  }else if(argc == 2){
    runFile(argv[1]);
  }else{
    fprintf(stderr, "USAGE PATH [CLOX] \n");
  }
  freeVM();
  

  return 0;
}