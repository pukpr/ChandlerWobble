generic
   Key : Integer;
   type Data_Type is private;
package Pace.Keyed_Shared_Memory.Io is
   type Data_Block is access Data_Type;
   pragma Warnings (Off);
   Pool : Block (Key);
   pragma Warnings (On);
   for Data_Block'Storage_Pool use Pool;

   -- simply read and write using Value
   Value : Data_Block := new Data_Type;
end Pace.Keyed_Shared_Memory.Io;
