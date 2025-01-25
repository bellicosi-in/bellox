#ifndef clox_object_h
#define clox_object_h

#include "value.h"
#include "chunk.h"

#define OBJ_TYPE(value) (AS_OBJ(value)->type)

#define  IS_STRING(value) isObjType(value, OBJ_STRING)
#define IS_FUNCTION(value)  isObjType(value, OBJ_FUNCTION)
#define IS_CLOSURE(value)      isObjType(value, OBJ_CLOSURE)

#define AS_CLOSURE(value) ((ObjClosure*)AS_OBJ(value))
#define AS_FUNCTION(value)  ((ObjFunction*)AS_OBJ(value))
#define AS_STRING(value) ((ObjString*)AS_OBJ(value))
#define AS_CSTRING(value) (((ObjString*)AS_OBJ(value))->chars)

typedef enum{
  OBJ_STRING,
  OBJ_FUNCTION,
  OBJ_CLOSURE,
  OBJ_UPVALUE,
}ObjType;

struct Obj{
  ObjType type;
  bool isMarked;
  struct Obj* next;
};

typedef struct{
  Obj obj;
  ObjString* name; //name of the function
  int arity; //parameters
  int upvalueCount;
  Chunk chunk; //chunk of the function

}ObjFunction;

struct ObjString{
  Obj obj;
  char* chars;
  int length;
  uint32_t hash;
};

typedef struct ObjUpvalue {
  Obj obj;
  Value* location;
  Value closed;
  struct ObjUpvalue* next;
} ObjUpvalue;

typedef struct{
  Obj obj;
  ObjFunction* function;
  ObjUpvalue** upvalues;
  int upvalueCount;
}ObjClosure;

ObjClosure* newClosure(ObjFunction* function);
ObjFunction* newFunction();
ObjString* copyString(const char* chars, int length);
ObjUpvalue* newUpvalue(Value* slot);
ObjString* takeString(char* chars, int length);
void printObject(Value value);

static inline bool isObjType(Value value, ObjType type){
  return IS_OBJ(value) && OBJ_TYPE(value)== type;
}


#endif