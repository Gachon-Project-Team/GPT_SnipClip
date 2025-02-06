import React from 'react'
import { Container } from '../components/Styles/Container/Container.style'
import { Header, MainHeader } from '../components/Styles/Header/Header.style'
import { Title } from '../components/Styles/Contents/Title.style'


export const Home = () => {
  return (
    <Container>
      <Header>
        <MainHeader>
          <Title>최신 뉴스를 간편하게 영상으로 !!</Title>
        </MainHeader>
      </Header>
    </Container>
  )
}
