#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <unistd.h>
#include <math.h>
#include "shared_structs.h"

int main() {
    // 1. Create the segment
    int shmid = shmget(SHM_KEY, sizeof(SharedWorldTable), IPC_CREAT | 0666);
    if (shmid < 0) {
        perror("shmget");
        exit(1);
    }

    // 2. Attach to the segment
    SharedWorldTable* shm = (SharedWorldTable*)shmat(shmid, NULL, 0);
    if (shm == (void*)-1) {
        perror("shmat");
        exit(1);
    }

    memset(shm, 0, sizeof(SharedWorldTable));
    shm->active_entities = 1;
    strncpy(shm->entities[0].name, "chassis", NAME_LEN);

    printf("Producer Active. System V SHM Key: %d\n", SHM_KEY);

    double t = 0;
    while (1) {
        shm->entities[0].x = 3.0 * cos(t);
        shm->entities[0].y = 3.0 * sin(t);
        shm->entities[0].z = 1.0;
        shm->entities[0].sequence++;

        t += 0.02;
        usleep(10000); 
    }
    return 0;
}
