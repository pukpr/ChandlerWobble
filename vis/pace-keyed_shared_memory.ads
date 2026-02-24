with System.Storage_Pools;
with System.Storage_Elements;

package Pace.Keyed_Shared_Memory is

   --------------------------------------------
   -- KEYED_SHARED_MEMORY -- IPC Shared Memory
   --------------------------------------------
   --
   -- Uses Ada Storage Pools to attach to Shared Memory
   -- example:
   --
   -- with Pace.Keyed_Shared_Memory;
   -- package Share is
   --   type Data is
   --     record
   --        Int : Integer;
   --        Flt : Float;
   --     end record;
   --   type Data_Ptr is access Data;
   --
   --   Pool : Pace.Keyed_Shared_Memory.Block (Key => 700, Size => 100);
   --   for Data_Ptr'Storage_Pool use Pool;
   -- end Share;

   use System.Storage_Pools;
   use System.Storage_Elements;

   type Block (Key : Integer) is new Root_Storage_Pool with private;

   procedure Allocate
     (Pool                     : in out Block;
      Storage_Address          : out System.Address;
      Size_In_Storage_Elements : in System.Storage_Elements.Storage_Count;
      Alignment                : in System.Storage_Elements.Storage_Count);

   procedure Deallocate
     (Pool                     : in out Block;
      Storage_Address          : in System.Address;
      Size_In_Storage_Elements : in System.Storage_Elements.Storage_Count;
      Alignment                : in System.Storage_Elements.Storage_Count);

   function Storage_Size
     (Pool : Block)
      return System.Storage_Elements.Storage_Count;

   Memory_Attach_Error : exception;

private

   type Block (Key : Integer) is new Root_Storage_Pool with record
      Shm_Id : Integer;
      Size   : System.Storage_Elements.Storage_Count;
   end record;

   ----------------------------------------------------------------------------
   ----
   -- $id: pace-keyed_shared_memory.ads,v 1.1 09/16/2002 18:18:26 pukitepa Exp
   --$
   ----------------------------------------------------------------------------
   ----
end Pace.Keyed_Shared_Memory;
