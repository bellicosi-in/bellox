#include "common.h"
#include "chunk.h"
#include "value.h"
#include "debug.h"
#include "memory.h"
#include "vm.h"


int main(int argc, char* argv[]){
  initVM();
  Chunk chunk;
  initChunk(&chunk);
  int constant = addConstant(&chunk, 1.2);
  writeChunk(&chunk, OP_CONSTANT, 123);
  writeChunk(&chunk, constant, 123);
  constant = addConstant(&chunk, 2.0);
  writeChunk(&chunk, OP_CONSTANT, 123);
  writeChunk(&chunk, constant, 123);
  writeChunk(&chunk, OP_ADD, 124);
  constant = addConstant(&chunk, 1.6);
  writeChunk(&chunk, OP_CONSTANT, 124);
  writeChunk(&chunk, constant, 124);
  writeChunk(&chunk, OP_DIVIDE, 124);
  writeChunk(&chunk, OP_NEGATE, 125);
  writeChunk(&chunk, OP_RETURN, 125);
  disassembleChunk(&chunk, "Test chunk");
  interpret(&chunk);
  freeVM();
  freeChunk(&chunk);
  

  return 0;
}