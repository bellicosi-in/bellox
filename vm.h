#ifndef clox_vm_h
#define clox_vm_h

#include "common.h"
#include "chunk.h"
#include "value.h"
#include "debug.h"
#include "table.h"
#include "object.h"

#define FRAMES_MAX 64
#define STACK_MAX (FRAMES_MAX * UINT8_COUNT)

//represents a single ongoing function call
typedef struct{
  ObjClosure* closure; //closure->function of the callframe
  uint8_t* ip; //base pointer where the caller will return
  Value* slots; //stack access
}CallFrame;

typedef struct{
  CallFrame frames[FRAMES_MAX];
  int frameCount;
  Value stack[STACK_MAX];
  Value* stackTop;
  Obj* objects;
  int grayCount;
  int grayCapacity;
  Obj** grayStack;
  Table globals;
  Table strings;
  ObjString* initString;
  ObjUpvalue* openUpvalues;
  size_t bytesAllocated;
  size_t nextGC;
}VM;

typedef enum{
  INTERPRET_COMPILE_ERROR,
  INTERPRET_RUNTIME_ERROR,
  INTERPRET_OK,
}InterpretResult;

extern VM vm;

void initVM();
void freeVM();
InterpretResult interpret(const char* source);
void push(Value value);
Value pop();

#endif