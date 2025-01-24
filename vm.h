#ifndef clox_vm_h
#define clox_vm_h

#include "common.h"
#include "chunk.h"
#include "value.h"
#include "debug.h"

#define STACK_MAX 256

typedef struct{
  uint8_t* ip;
  Chunk* chunk;
  Value stack[STACK_MAX];
  Value* stackTop;
}VM;

typedef enum{
  INTERPRET_COMPILE_ERROR,
  INTERPRET_RUNTIME_ERROR,
  INTERPRET_OK,
}InterpretResult;

extern VM vm;

void initVM();
void freeVM();
InterpretResult interpret(Chunk* chunk);
void push(Value value);
Value pop();

#endif