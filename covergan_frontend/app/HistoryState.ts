import { useCallback, useState } from "react";

type InitialValueType<T> = T | ((prev?: T) => T);

type ReturnValueType<T> = [
  T,
  (value: InitialValueType<T>) => void,
  () => void,
  () => void,
  InitialValueType<T>[],
  number
];

const useHistoryState: <T>(
  initialValue?: InitialValueType<T>
) => ReturnValueType<T> = <T>(initialValue?: InitialValueType<T>) => {
  const [state, _setState] = useState<T>(initialValue);
  const [history, setHistory] = useState<InitialValueType<T>[]>(
    initialValue !== undefined && initialValue !== null ? [initialValue] : []
  );
  const [pointer, setPointer] = useState<number>(
    initialValue !== undefined && initialValue !== null ? 0 : -1
  );

  const setState: (value: InitialValueType<T>) => void = useCallback(
    (value: InitialValueType<T>) => {
      let valueToAdd = value;
      if (typeof value === "function") {
        valueToAdd = (value as (prev?: T) => T)(state);
      } else if (pointer > 0) {
        if (typeof value === "object" &&
          typeof history[pointer] === "object" &&
          JSON.stringify(value) === JSON.stringify(history[pointer])) {
          return;
        }
      }
      setHistory([...history.slice(0, pointer + 1), valueToAdd]);
      setPointer(pointer + 1);
      _setState(value);
    },
    [setHistory, setPointer, _setState, state, pointer]
  );

  const undo: () => void = useCallback(() => {
    if (pointer <= 0) return;
    _setState(history[pointer - 1]);
    setPointer(pointer - 1);
  }, [history, pointer, setPointer]);

  const redo: () => void = useCallback(() => {
    if (pointer + 1 >= history.length) return;
    _setState(history[pointer + 1]);
    setPointer(pointer + 1);
  }, [pointer, history, setPointer]);

  return [state, setState, undo, redo, history, pointer];
};

export default useHistoryState;