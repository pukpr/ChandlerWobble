package body Pace.Keyed_Shared_Memory is
   ----------------------
   -- UNIX only body
   ----------------------

   Ipc_Alloc : constant := 8#0100000#;
   Ipc_Creat : constant := 8#0001000#;
   Shm_R     : constant := 8#0444#;
   Shm_W     : constant := 8#0222#;

   function Shmget (Key, Size, Flags : Integer) return Integer;
   pragma Import (C, Shmget, "shmget");

   function Shmat (Id, Address, Flags : Integer) return System.Address;
   pragma Import (C, Shmat, "shmat");

   function Shmdt (Addr : System.Address) return Integer;
   pragma Import (C, Shmdt, "shmdt");

   procedure Allocate
     (Pool                     : in out Block;
      Storage_Address          : out System.Address;
      Size_In_Storage_Elements : in System.Storage_Elements.Storage_Count;
      Alignment                : in System.Storage_Elements.Storage_Count)
   is

      use System.Storage_Elements;
      Id : Integer;
   begin
      Pool.Size := Size_In_Storage_Elements;
      Id        :=
         Shmget
           (Pool.Key,
            Integer (Pool.Size),
            Ipc_Alloc + Ipc_Creat + Shm_R + Shm_W);
      if Id < 0 then
         raise Memory_Attach_Error;
      end if;
      Pool.Shm_Id     := Id;
      Storage_Address := Shmat (Id, 0, 0);

   end Allocate;

   procedure Deallocate
     (Pool                     : in out Block;
      Storage_Address          : in System.Address;
      Size_In_Storage_Elements : in System.Storage_Elements.Storage_Count;
      Alignment                : in System.Storage_Elements.Storage_Count)
   is
      Ret : Integer;
   begin
      Ret := Shmdt (Storage_Address);
   end Deallocate;

   function Storage_Size
     (Pool : Block)
      return System.Storage_Elements.Storage_Count
   is
   begin
      return Pool.Size;
   end Storage_Size;

   ----------------------------------------------------------------------------
   ----
   -- $id: pace-keyed_shared_memory.adb,v 1.1 09/16/2002 18:18:26 pukitepa Exp
   --$
   ----------------------------------------------------------------------------
   ----
end Pace.Keyed_Shared_Memory;
