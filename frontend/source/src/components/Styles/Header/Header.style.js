import styled from "styled-components";

export const Header = styled.section`
    width: 100%;
    height: 80vh;
    position: relative;
    display: flex;
    justify-content: center;
    overflow: hidden;
    align-items: center;
    &::before {
        content: '';
        position: absolute;
        width: 100%;
        height: 100%;
        top: 0;
        left: 0;
        max-width: 100%;
        min-height: 40px;
        display: inline-block;
        border-radius : 0 0 70% 70% / 0 0 100% 100%;
        transform: scaleX(1.5);
        background-position: right top;
        background-size: 100vw 200px;
        background-color: #3C404B;
    }

  /* border: solid blue 2px; */
`

export const MainHeader = styled.div`
    position: relative;
    z-index: 1;
    margin: 0 auto;
    width: 100%;
    max-width: 1200px;
    background-color: transparent;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 20px;
  @media (max-width: 768px) {
    padding: 15px;
  }

  @media (max-width: 480px) {
    padding: 10px;
  }

  /* border: 2px solid blue; */
`