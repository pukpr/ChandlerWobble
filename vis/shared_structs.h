#ifndef SHARED_STRUCTS_H
#define SHARED_STRUCTS_H

#include <stdint.h>

#define SHM_KEY 123456  // Common integer key
#define MAX_ENTITIES 32
#define NAME_LEN 64

typedef struct {
    char name[NAME_LEN];
    int command;
    double x, y, z;
    double roll, pitch, yaw;
    volatile uint64_t sequence; 
} EntityState;

typedef struct {
    uint32_t active_entities;
    EntityState entities[MAX_ENTITIES];
} SharedWorldTable;

#endif
