#include "common.h"
#include "chunk.h"
#include "value.h"
#include "debug.h"
#include "memory.h"


int main(int argc, char* argv[]){
  Chunk chunk;
  initChunk(&chunk);
  int constant = addConstant(&chunk, 1.2);
  writeChunk(&chunk, OP_CONSTANT, 123);
  writeChunk(&chunk, constant, 123);
  writeChunk(&chunk, OP_RETURN, 124);
  disassembleChunk(&chunk, "Test chunk");
  freeChunk(&chunk);

  return 0;
}