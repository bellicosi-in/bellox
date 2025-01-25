#ifndef clox_object_h
#define clox_object_h

#include "value.h"
#include "chunk.h"

#define OBJ_TYPE(value) (AS_OBJ(value)->type)

#define  IS_STRING(value) isObjType(value, OBJ_STRING)
#define IS_FUNCTION(value)  isObjType(value, OBJ_FUNCTION)

#define AS_FUNCTION(value)  ((ObjFunction*)AS_OBJ(value))
#define AS_STRING(value) ((ObjString*)AS_OBJ(value))
#define AS_CSTRING(value) (((ObjString*)AS_OBJ(value))->chars)

typedef enum{
  OBJ_STRING,
  OBJ_FUNCTION,
}ObjType;

struct Obj{
  struct Obj* next;
  ObjType type;
};

typedef struct{
  Obj obj;
  ObjString* name; //name of the function
  int arity; //parameters
  Chunk chunk; //chunk of the function

}ObjFunction;

struct ObjString{
  Obj obj;
  char* chars;
  int length;
  uint32_t hash;
};

ObjFunction* newFunction();
ObjString* copyString(const char* chars, int length);
ObjString* takeString(char* chars, int length);
void printObject(Value value);

static inline bool isObjType(Value value, ObjType type){
  return IS_OBJ(value) && OBJ_TYPE(value)== type;
}


#endif