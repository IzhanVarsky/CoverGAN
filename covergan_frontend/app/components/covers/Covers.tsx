import { useOutletContext } from '@remix-run/react';
import Carousel from './Carousel';
import Viewer from './Viewer';
import { Text, Center } from '@mantine/core';
import Shape from '../Shape';

export default function Covers() {
  const [selectedCover, setSelectedCover, covers, setCovers] = useOutletContext();

  return (
    covers.length > 0 ?
      <>
        <Carousel/>
        <Viewer/>
      </>
      :
      <Shape>
        <Center>
          <Text size={'xl'} weight={'bolder'}>Hi! Generate some pictures!</Text>
        </Center>
      </Shape>
  )
}