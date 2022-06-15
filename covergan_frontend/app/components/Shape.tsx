import { Paper } from '@mantine/core';

export default function Shape({ children, style }) {
  return (
    <Paper withBorder radius='lg' shadow="xl" p="xl" m='md' style={style}>
      {children}
    </Paper>
  )
}