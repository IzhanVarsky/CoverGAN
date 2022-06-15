import { Grid, Image, AspectRatio, Container, Center } from '@mantine/core';
import Shape from '../Shape';
import { useOutletContext } from "@remix-run/react";

export default function Carousel() {
  const [selectedCover, setSelectedCover, covers, setCovers] = useOutletContext();
  return (
    <Shape>
      <Grid justify="space-around" columns={covers.length}>
        {covers.map((cover, index) => {
          return (
            <Grid.Col span={1} key={index}>
              <Center>
                <Image
                  key={index}
                  src={'data:image/png;base64, ' + cover.base64}
                  onClick={() => setSelectedCover(index)}
                  style={{
                    outline: index == selectedCover ? "5px solid #228be6" : "0",
                    cursor: 'pointer',
                    maxWidth: '16vh'
                  }}
                />
              </Center>
            </Grid.Col>
          )
        })}
      </Grid>
    </Shape>
  )
}